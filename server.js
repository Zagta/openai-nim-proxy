// server.js - OpenAI to NVIDIA NIM API Proxy (Chub-friendly)
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '40mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Show/hide reasoning in output
const SHOW_REASONING = false;

// DeepSeek-V3.2 thinking mode
const ENABLE_THINKING_MODE = true;

// Timeouts / retry / logging config
const NIM_TIMEOUT_MS = Number(process.env.NIM_TIMEOUT_MS || 180000);
const NIM_RESPONSE_HEADERS_TIMEOUT_MS = Number(process.env.NIM_RESPONSE_HEADERS_TIMEOUT_MS || 15000);
const NIM_FIRST_CHUNK_TIMEOUT_MS = Number(process.env.NIM_FIRST_CHUNK_TIMEOUT_MS || 30000);
const NIM_STREAM_IDLE_TIMEOUT_MS = Number(process.env.NIM_STREAM_IDLE_TIMEOUT_MS || 20000);

const NIM_TIMEOUT_RETRY_SWITCH = /^(1|true|yes|on)$/i.test(
  String(process.env.NIM_TIMEOUT_RETRY_SWITCH || 'true')
);
const NIM_TIMEOUT_MAX_RETRIES = Number(process.env.NIM_TIMEOUT_MAX_RETRIES || 5);
const NIM_TIMEOUT_RETRY_DELAY_MS = Number(process.env.NIM_TIMEOUT_RETRY_DELAY_MS || 1500);

const SLOW_REQUEST_MS = Number(process.env.SLOW_REQUEST_MS || 15000);

const RECENT_REQUESTS_LIMIT = 5;
const recentRequests = [];

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4o': 'qwen/qwen3-coder-480b-a35b-instruct',

  // DeepSeek aliases
  'gpt-4': 'deepseek-ai/deepseek-v3.2',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.2-speciale',

  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

const DEBUG_RECEIVED_OPTION_KEYS = [
  'stream',
  'max_tokens',
  'max_completion_tokens',
  'temperature',
  'top_p',
  'top_k',
  'repetition_penalty',
  'presence_penalty',
  'frequency_penalty',
  'stop',
  'seed',
  'n',
  'user'
];

const NIM_FORWARD_OPTION_KEYS = [
  'temperature',
  'top_p',
  'top_k',
  'repetition_penalty',
  'presence_penalty',
  'frequency_penalty',
  'stop',
  'seed',
  'n',
  'user'
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function contentToString(content) {
  if (content == null) return '';

  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part == null) return '';

        if (typeof part === 'string') {
          return part;
        }

        if (typeof part === 'object') {
          if ((part.type === 'text' || part.type === 'input_text') && typeof part.text === 'string') {
            return part.text;
          }

          if (typeof part.text === 'string') {
            return part.text;
          }

          if (typeof part.content === 'string') {
            return part.content;
          }

          if (Array.isArray(part.content)) {
            return contentToString(part.content);
          }
        }

        return '';
      })
      .join('');
  }

  if (typeof content === 'object') {
    if (typeof content.text === 'string') {
      return content.text;
    }

    if (Array.isArray(content.content)) {
      return contentToString(content.content);
    }

    if (typeof content.content === 'string') {
      return content.content;
    }

    try {
      return JSON.stringify(content);
    } catch {
      return String(content);
    }
  }

  return String(content);
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return [];

  return messages.map((msg) => {
    const role = msg?.role === 'developer' ? 'system' : msg?.role;

    const normalized = {
      ...msg,
      role,
      content: contentToString(msg?.content)
    };

    if (normalized.content == null) {
      normalized.content = '';
    }

    return normalized;
  });
}

function pickDefined(source, keys) {
  const out = {};
  for (const key of keys) {
    if (source[key] !== undefined) {
      out[key] = source[key];
    }
  }
  return out;
}

function sanitizeDebugValue(key, value) {
  if (value === undefined) return undefined;

  if (key === 'user') {
    return value == null ? value : '[present]';
  }

  if (key === 'stop') {
    if (typeof value === 'string') {
      return {
        type: 'string',
        length: value.length
      };
    }

    if (Array.isArray(value)) {
      return {
        type: 'array',
        count: value.length,
        item_lengths: value.map((v) => String(v ?? '').length)
      };
    }

    return '[present]';
  }

  return value;
}

function sanitizeDebugObject(source, keys) {
  const out = {};
  for (const key of keys) {
    if (source[key] !== undefined) {
      out[key] = sanitizeDebugValue(key, source[key]);
    }
  }
  return out;
}

function safeBodyKeys(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return [];
  }

  return Object.keys(body).sort();
}

function createOpenAIError(status, message, type = 'invalid_request_error') {
  return {
    error: {
      message,
      type,
      code: status
    }
  };
}

function addRecentRequest(entry) {
  recentRequests.unshift(entry);
  if (recentRequests.length > RECENT_REQUESTS_LIMIT) {
    recentRequests.length = RECENT_REQUESTS_LIMIT;
  }
}

function updateRecentRequest(id, patch) {
  const item = recentRequests.find((r) => r.id === id);
  if (!item) return;
  Object.assign(item, patch);
}

function finalizeRecentRequest(id, patch) {
  updateRecentRequest(id, {
    ...patch,
    finished_at: new Date().toISOString()
  });
}

function getRequestLogById(id) {
  return recentRequests.find((r) => r.id === id) || null;
}

function logRequestSummary(record) {
  const base = `[${record.id}] ${record.status} model=${record.client_model} -> ${record.nim_model || 'n/a'} duration=${record.duration_ms ?? 'n/a'}ms attempts=${record.attempt_count ?? 1} winner=${record.winner_attempt ?? 'n/a'}`;
  if ((record.duration_ms ?? 0) >= SLOW_REQUEST_MS) {
    console.warn('[slow-request]', base);
  } else {
    console.log('[request]', base);
  }
}

function safeDestroyStream(stream) {
  try {
    if (stream?.destroy) {
      stream.destroy();
    }
  } catch {
    // ignore
  }
}

function isAbortLikeError(error) {
  return error?.code === 'ERR_CANCELED' || error?.name === 'CanceledError' || error?.name === 'AbortError';
}

function openAIErrorTypeByStatus(status, fallback = 'api_error') {
  if (status >= 500) return 'server_error';
  if (status >= 400) return 'invalid_request_error';
  return fallback;
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    nim_api_base: NIM_API_BASE,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    api_key_configured: Boolean(NIM_API_KEY),
    nim_timeout_ms: NIM_TIMEOUT_MS,
    nim_response_headers_timeout_ms: NIM_RESPONSE_HEADERS_TIMEOUT_MS,
    nim_first_chunk_timeout_ms: NIM_FIRST_CHUNK_TIMEOUT_MS,
    nim_stream_idle_timeout_ms: NIM_STREAM_IDLE_TIMEOUT_MS,
    nim_timeout_retry_switch: NIM_TIMEOUT_RETRY_SWITCH,
    nim_timeout_max_retries: NIM_TIMEOUT_MAX_RETRIES,
    nim_timeout_retry_delay_ms: NIM_TIMEOUT_RETRY_DELAY_MS,
    slow_request_ms: SLOW_REQUEST_MS,
    recent_requests_kept: RECENT_REQUESTS_LIMIT
  });
});

// Safe recent requests debug endpoint
app.get('/debug/recent-requests', (req, res) => {
  const data = recentRequests.map((r) => ({
    started_at: r.started_at,
    finished_at: r.finished_at || null,
    status: r.status,
    client_model: r.client_model,
    nim_model: r.nim_model,
    stream: r.stream,
    retry_enabled: Boolean(r.retry_enabled),
    retry_max_attempts: r.retry_max_attempts ?? null,
    retry_delay_ms: r.retry_delay_ms ?? null,
    message_count: r.message_count,
    body_keys: r.body_keys || [],
    requested_max_tokens: r.requested_max_tokens,
    request_options_received: r.request_options_received || {},
    request_options_forwarded: r.request_options_forwarded || {},
    model_probe_used: Boolean(r.model_probe_used),
    nim_status: r.nim_status ?? null,
    first_byte_ms: r.first_byte_ms ?? null,
    duration_ms: r.duration_ms ?? null,
    choice_count: r.choice_count ?? null,
    usage: r.usage || null,
    attempt_count: r.attempt_count ?? 0,
    winner_attempt: r.winner_attempt ?? null,
    timed_out_stage: r.timed_out_stage ?? null,
    attempts: r.attempts || [],
    has_error: Boolean(r.error),
    error_type:
      r.status === 'timeout'
        ? 'timeout'
        : r.status === 'stream_error'
          ? 'stream_error'
          : r.status === 'error'
            ? 'error'
            : r.status === 'configuration_error'
              ? 'configuration_error'
              : r.status === 'rejected'
                ? 'rejected'
                : null
  }));

  res.json({
    limit: RECENT_REQUESTS_LIMIT,
    count: data.length,
    data
  });
});

// OpenAI-compatible models list
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Main proxy endpoint
app.post('/v1/chat/completions', async (req, res) => {
  const body = req.body || {};
  const requestId = `req_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const startedAt = Date.now();
  const downstreamStreamRequested = Boolean(body.stream);

  addRecentRequest({
    id: requestId, // internal only, not returned by debug endpoint
    started_at: new Date(startedAt).toISOString(),
    status: 'received',
    client_model: body.model || null,
    nim_model: null,
    stream: downstreamStreamRequested,
    retry_enabled: NIM_TIMEOUT_RETRY_SWITCH,
    retry_max_attempts: NIM_TIMEOUT_MAX_RETRIES,
    retry_delay_ms: NIM_TIMEOUT_RETRY_DELAY_MS,
    message_count: Array.isArray(body.messages) ? body.messages.length : 0,
    body_keys: safeBodyKeys(body),
    requested_max_tokens:
      typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : 64000,
    request_options_received: sanitizeDebugObject(body, DEBUG_RECEIVED_OPTION_KEYS),
    attempt_count: 0,
    winner_attempt: null,
    attempts: []
  });

  const state = {
    finalized: false,
    clientClosed: false,
    clientStreamRequested: downstreamStreamRequested,
    clientHeadersSent: false,
    winnerAttempt: null,
    attempts: [],
    activeController: null,
    activeStream: null,
    reasoningStarted: false,
    aggregate: {
      id: null,
      created: null,
      usage: null,
      finishReason: 'stop',
      role: 'assistant',
      content: '',
      reasoning: ''
    }
  };

  function remainingAbsoluteMs() {
    return NIM_TIMEOUT_MS - (Date.now() - startedAt);
  }

  function snapshotAttempts() {
    return state.attempts.map((attempt) => ({
      index: attempt.index,
      status: attempt.status,
      headers_ms: attempt.headersAt ? attempt.headersAt - startedAt : null,
      first_byte_ms: attempt.firstDataAt ? attempt.firstDataAt - startedAt : null,
      nim_status: attempt.nimStatus ?? null,
      error_type:
        attempt.status === 'aborted'
          ? 'aborted'
          : attempt.status === 'error'
            ? 'error'
            : attempt.status === 'stream_error'
              ? 'stream_error'
              : null
    }));
  }

  function updateAttemptsDebug() {
    updateRecentRequest(requestId, {
      attempt_count: state.attempts.length,
      winner_attempt: state.winnerAttempt,
      attempts: snapshotAttempts()
    });
  }

  function markClientClosed() {
    if (state.finalized || res.writableEnded) return;

    state.clientClosed = true;
    state.finalized = true;

    try {
      state.activeController?.abort();
    } catch {
      // ignore
    }

    safeDestroyStream(state.activeStream);

    finalizeRecentRequest(requestId, {
      status: 'client_closed',
      duration_ms: Date.now() - startedAt,
      attempt_count: state.attempts.length,
      winner_attempt: state.winnerAttempt
    });
  }

  function ensureClientStreamHeaders() {
    if (!state.clientStreamRequested || state.clientHeadersSent) return;

    state.clientHeadersSent = true;
    res.status(200);
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    if (typeof res.flushHeaders === 'function') {
      res.flushHeaders();
    }
  }

  function buildOpenAIResponseFromAggregate(model) {
    let fullContent = state.aggregate.content;

    if (SHOW_REASONING && state.aggregate.reasoning) {
      fullContent = `<think>\n${state.aggregate.reasoning}\n</think>\n\n${fullContent}`;
    }

    return {
      id: state.aggregate.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: state.aggregate.created || Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          message: {
            role: state.aggregate.role || 'assistant',
            content: fullContent
          },
          finish_reason: state.aggregate.finishReason || 'stop'
        }
      ],
      usage: state.aggregate.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };
  }

  function finalizeSuccess(model, nimModel) {
    if (state.finalized) return;
    state.finalized = true;

    const durationMs = Date.now() - startedAt;
    const winner = state.attempts.find((a) => a.index === state.winnerAttempt) || null;

    finalizeRecentRequest(requestId, {
      status: 'completed',
      duration_ms: durationMs,
      nim_status: winner?.nimStatus ?? null,
      first_byte_ms: winner?.firstDataAt ? winner.firstDataAt - startedAt : null,
      usage: state.aggregate.usage || null,
      choice_count: 1,
      attempt_count: state.attempts.length,
      winner_attempt: state.winnerAttempt
    });

    logRequestSummary(getRequestLogById(requestId) || {
      id: requestId,
      status: 'completed',
      client_model: model,
      nim_model: nimModel,
      duration_ms: durationMs,
      attempt_count: state.attempts.length,
      winner_attempt: state.winnerAttempt
    });

    if (state.clientStreamRequested) {
      if (!res.writableEnded) {
        res.end();
      }
      return;
    }

    res.json(buildOpenAIResponseFromAggregate(model));
  }

  function finalizeWithError(model, nimModel, status, message, errorStatus = 'error', timedOutStage = null) {
    if (state.finalized) return;
    state.finalized = true;

    try {
      state.activeController?.abort();
    } catch {
      // ignore
    }

    safeDestroyStream(state.activeStream);

    const durationMs = Date.now() - startedAt;
    const winner = state.attempts.find((a) => a.index === state.winnerAttempt) || null;

    finalizeRecentRequest(requestId, {
      status: errorStatus,
      duration_ms: durationMs,
      nim_status: winner?.nimStatus ?? null,
      first_byte_ms: winner?.firstDataAt ? winner.firstDataAt - startedAt : null,
      error: message,
      timed_out_stage: timedOutStage,
      attempt_count: state.attempts.length,
      winner_attempt: state.winnerAttempt
    });

    if (state.clientStreamRequested && state.clientHeadersSent) {
      res.end();
      return;
    }

    res
      .status(status)
      .json(
        createOpenAIError(
          status,
          message,
          errorStatus === 'timeout' ? 'timeout_error' : openAIErrorTypeByStatus(status)
        )
      );

    logRequestSummary(getRequestLogById(requestId) || {
      id: requestId,
      status: errorStatus,
      client_model: model,
      nim_model: nimModel,
      duration_ms: durationMs,
      attempt_count: state.attempts.length,
      winner_attempt: state.winnerAttempt
    });
  }

  function processWinnerLine(line) {
    if (!line.startsWith('data: ')) return;

    const payload = line.slice(6);

    if (payload.includes('[DONE]')) {
      if (state.clientStreamRequested) {
        ensureClientStreamHeaders();
        res.write('data: [DONE]\n\n');
      }
      return;
    }

    try {
      const data = JSON.parse(payload);

      if (data.id && !state.aggregate.id) {
        state.aggregate.id = data.id;
      }

      if (data.created && !state.aggregate.created) {
        state.aggregate.created = data.created;
      }

      if (data.usage) {
        state.aggregate.usage = data.usage;
      }

      const choice = data.choices?.[0];
      if (choice?.finish_reason) {
        state.aggregate.finishReason = choice.finish_reason;
      }

      if (choice?.delta) {
        const delta = choice.delta;
        const reasoning = contentToString(delta.reasoning_content);
        const content = contentToString(delta.content);

        if (reasoning) {
          state.aggregate.reasoning += reasoning;
        }

        if (content) {
          state.aggregate.content += content;
        }

        if (choice.delta.role) {
          state.aggregate.role = choice.delta.role;
        }

        if (state.clientStreamRequested) {
          if (SHOW_REASONING) {
            let combinedContent = '';

            if (reasoning && !state.reasoningStarted) {
              combinedContent += '<think>\n' + reasoning;
              state.reasoningStarted = true;
            } else if (reasoning) {
              combinedContent += reasoning;
            }

            if (content && state.reasoningStarted) {
              combinedContent += '</think>\n\n' + content;
              state.reasoningStarted = false;
            } else if (content) {
              combinedContent += content;
            }

            delta.content = combinedContent || '';
            delete delta.reasoning_content;
          } else {
            delta.content = content || '';
            delete delta.reasoning_content;
          }

          ensureClientStreamHeaders();
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
      } else if (state.clientStreamRequested) {
        ensureClientStreamHeaders();
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      }
    } catch {
      if (state.clientStreamRequested) {
        ensureClientStreamHeaders();
        res.write(line + '\n\n');
      }
    }
  }

  async function runAttemptUntilFirstChunk(index, nimRequest) {
    const attempt = {
      index,
      controller: new AbortController(),
      response: null,
      startedAt: Date.now(),
      headersAt: null,
      firstDataAt: null,
      nimStatus: null,
      status: 'starting',
      error: null,
      aborted: false,
      abortReason: null,
      settled: false
    };

    state.attempts.push(attempt);
    state.activeController = attempt.controller;
    updateAttemptsDebug();

    const absoluteRemainingAtStart = remainingAbsoluteMs();
    if (absoluteRemainingAtStart <= 0) {
      attempt.status = 'aborted';
      attempt.aborted = true;
      attempt.abortReason = 'timeout_absolute';
      attempt.settled = true;
      updateAttemptsDebug();

      return {
        type: 'timeout',
        stage: 'absolute',
        retryable: false,
        message: `Upstream absolute timeout after ${NIM_TIMEOUT_MS} ms`,
        attempt
      };
    }

    const headerTimeoutMs = Math.max(
      1,
      Math.min(NIM_RESPONSE_HEADERS_TIMEOUT_MS, absoluteRemainingAtStart)
    );

    try {
      const response = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: 'stream',
          timeout: headerTimeoutMs,
          signal: attempt.controller.signal
        }
      );

      if (state.clientClosed || state.finalized) {
        safeDestroyStream(response.data);
        return { type: 'client_closed', attempt };
      }

      attempt.response = response;
      attempt.nimStatus = response.status;
      attempt.headersAt = Date.now();
      attempt.status = 'headers_received';
      state.activeStream = response.data;
      updateAttemptsDebug();

      const absoluteRemainingForFirstChunk = remainingAbsoluteMs();
      if (absoluteRemainingForFirstChunk <= 0) {
        attempt.status = 'aborted';
        attempt.aborted = true;
        attempt.abortReason = 'timeout_absolute';
        attempt.settled = true;
        safeDestroyStream(response.data);
        updateAttemptsDebug();

        return {
          type: 'timeout',
          stage: 'absolute',
          retryable: false,
          message: `Upstream absolute timeout after ${NIM_TIMEOUT_MS} ms`,
          attempt
        };
      }

      const firstChunkTimeoutMs = Math.max(
        1,
        Math.min(NIM_FIRST_CHUNK_TIMEOUT_MS, absoluteRemainingForFirstChunk)
      );

      return await new Promise((resolve) => {
        let settled = false;
        const stream = response.data;

        const cleanup = () => {
          clearTimeout(firstChunkTimer);
          stream.off('data', onData);
          stream.off('end', onEnd);
          stream.off('error', onError);
        };

        const firstChunkTimer = setTimeout(() => {
          if (settled) return;
          settled = true;

          cleanup();

          attempt.status = 'aborted';
          attempt.aborted = true;
          attempt.abortReason =
            firstChunkTimeoutMs < NIM_FIRST_CHUNK_TIMEOUT_MS
              ? 'timeout_absolute'
              : 'timeout_before_first_chunk';
          attempt.settled = true;

          try {
            attempt.controller.abort();
          } catch {
            // ignore
          }

          safeDestroyStream(stream);
          updateAttemptsDebug();

          const stage = attempt.abortReason === 'timeout_absolute' ? 'absolute' : 'before_first_chunk';

          resolve({
            type: 'timeout',
            stage,
            retryable: stage !== 'absolute',
            message:
              stage === 'absolute'
                ? `Upstream absolute timeout after ${NIM_TIMEOUT_MS} ms`
                : `No first stream chunk after ${NIM_FIRST_CHUNK_TIMEOUT_MS} ms`,
            attempt
          });
        }, firstChunkTimeoutMs);

        const onData = (chunk) => {
          if (settled) return;
          settled = true;

          cleanup();

          attempt.firstDataAt = Date.now();
          attempt.status = 'streaming';
          updateAttemptsDebug();

          stream.pause();

          resolve({
            type: 'first_chunk',
            attempt,
            response,
            firstChunk: chunk
          });
        };

        const onEnd = () => {
          if (settled) return;
          settled = true;

          cleanup();

          attempt.status = 'completed';
          attempt.settled = true;
          updateAttemptsDebug();

          resolve({
            type: 'ended_before_first_chunk',
            attempt,
            message: 'Upstream stream ended before first chunk'
          });
        };

        const onError = (error) => {
          if (settled) return;
          settled = true;

          cleanup();

          attempt.settled = true;

          if (state.clientClosed) {
            attempt.status = 'aborted';
            attempt.aborted = true;
            attempt.abortReason = 'client_closed';
            updateAttemptsDebug();

            resolve({ type: 'client_closed', attempt });
            return;
          }

          if (attempt.abortReason === 'timeout_before_first_chunk') {
            attempt.status = 'aborted';
            attempt.aborted = true;
            updateAttemptsDebug();

            resolve({
              type: 'timeout',
              stage: 'before_first_chunk',
              retryable: true,
              message: `No first stream chunk after ${NIM_FIRST_CHUNK_TIMEOUT_MS} ms`,
              attempt
            });
            return;
          }

          if (attempt.abortReason === 'timeout_absolute') {
            attempt.status = 'aborted';
            attempt.aborted = true;
            updateAttemptsDebug();

            resolve({
              type: 'timeout',
              stage: 'absolute',
              retryable: false,
              message: `Upstream absolute timeout after ${NIM_TIMEOUT_MS} ms`,
              attempt
            });
            return;
          }

          if (isAbortLikeError(error)) {
            attempt.status = 'aborted';
            attempt.aborted = true;
            attempt.abortReason = attempt.abortReason || 'aborted';
            updateAttemptsDebug();

            resolve({
              type: attempt.abortReason === 'client_closed' ? 'client_closed' : 'aborted',
              attempt
            });
            return;
          }

          attempt.status = 'error';
          attempt.error = error;
          updateAttemptsDebug();

          resolve({
            type: 'error_before_first_chunk',
            attempt,
            error
          });
        };

        stream.on('data', onData);
        stream.on('end', onEnd);
        stream.on('error', onError);
      });
    } catch (error) {
      attempt.settled = true;

      if (state.clientClosed) {
        attempt.status = 'aborted';
        attempt.aborted = true;
        attempt.abortReason = 'client_closed';
        updateAttemptsDebug();

        return { type: 'client_closed', attempt };
      }

      if (error.code === 'ECONNABORTED') {
        attempt.status = 'aborted';
        attempt.aborted = true;
        attempt.error = error;

        const stage =
          headerTimeoutMs < NIM_RESPONSE_HEADERS_TIMEOUT_MS
            ? 'absolute'
            : 'response_headers';

        updateAttemptsDebug();

        return {
          type: 'timeout',
          stage,
          retryable: stage !== 'absolute',
          message:
            stage === 'absolute'
              ? `Upstream absolute timeout after ${NIM_TIMEOUT_MS} ms`
              : `Upstream response headers timeout after ${NIM_RESPONSE_HEADERS_TIMEOUT_MS} ms`,
          attempt
        };
      }

      if (isAbortLikeError(error)) {
        attempt.status = 'aborted';
        attempt.aborted = true;
        attempt.abortReason = attempt.abortReason || 'aborted';
        updateAttemptsDebug();

        return {
          type: state.clientClosed ? 'client_closed' : 'aborted',
          attempt
        };
      }

      if (error.response?.status) {
        attempt.status = 'error';
        attempt.error = error;
        updateAttemptsDebug();

        return {
          type: 'http_error',
          attempt,
          error
        };
      }

      attempt.status = 'error';
      attempt.error = error;
      updateAttemptsDebug();

      return {
        type: 'error_before_first_chunk',
        attempt,
        error
      };
    }
  }

  async function consumeWinningStream(model, nimModel, attempt, response, firstChunk) {
    state.winnerAttempt = attempt.index;
    state.activeController = attempt.controller;
    state.activeStream = response.data;

    updateRecentRequest(requestId, {
      status: 'streaming',
      nim_status: attempt.nimStatus,
      first_byte_ms: attempt.firstDataAt ? attempt.firstDataAt - startedAt : null,
      winner_attempt: attempt.index
    });

    updateAttemptsDebug();

    const stream = response.data;
    let buffer = '';

    const processChunk = (chunk) => {
      buffer += chunk.toString('utf8');
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        processWinnerLine(line);
      }
    };

    return await new Promise((resolve) => {
      let idleTimer = null;

      const cleanup = () => {
        clearTimeout(idleTimer);
        stream.off('data', onData);
        stream.off('end', onEnd);
        stream.off('error', onError);
      };

      const resetIdleTimer = () => {
        clearTimeout(idleTimer);

        const remaining = remainingAbsoluteMs();
        if (remaining <= 0) {
          attempt.status = 'aborted';
          attempt.aborted = true;
          attempt.abortReason = 'timeout_absolute';
          attempt.settled = true;
          cleanup();

          try {
            attempt.controller.abort();
          } catch {
            // ignore
          }

          safeDestroyStream(stream);
          updateAttemptsDebug();

          resolve({
            type: 'absolute_timeout',
            attempt
          });
          return;
        }

        const effectiveIdleMs = Math.max(
          1,
          Math.min(NIM_STREAM_IDLE_TIMEOUT_MS, remaining)
        );

        idleTimer = setTimeout(() => {
          attempt.status = 'aborted';
          attempt.aborted = true;
          attempt.abortReason =
            effectiveIdleMs < NIM_STREAM_IDLE_TIMEOUT_MS
              ? 'timeout_absolute'
              : 'stream_idle';
          attempt.settled = true;

          cleanup();

          try {
            attempt.controller.abort();
          } catch {
            // ignore
          }

          safeDestroyStream(stream);
          updateAttemptsDebug();

          resolve({
            type:
              attempt.abortReason === 'timeout_absolute'
                ? 'absolute_timeout'
                : 'stream_idle_timeout',
            attempt
          });
        }, effectiveIdleMs);
      };

      const onData = (chunk) => {
        if (state.clientClosed) {
          cleanup();
          resolve({ type: 'client_closed', attempt });
          return;
        }

        resetIdleTimer();
        processChunk(chunk);
        updateAttemptsDebug();
      };

      const onEnd = () => {
        cleanup();

        if (buffer.trim().length > 0) {
          processWinnerLine(buffer.trimEnd());
          buffer = '';
        }

        attempt.status = 'completed';
        attempt.settled = true;
        updateAttemptsDebug();

        resolve({
          type: 'completed',
          attempt
        });
      };

      const onError = (error) => {
        cleanup();
        attempt.settled = true;

        if (state.clientClosed) {
          attempt.status = 'aborted';
          attempt.aborted = true;
          attempt.abortReason = 'client_closed';
          updateAttemptsDebug();

          resolve({ type: 'client_closed', attempt });
          return;
        }

        if (attempt.abortReason === 'stream_idle') {
          attempt.status = 'aborted';
          attempt.aborted = true;
          updateAttemptsDebug();

          resolve({
            type: 'stream_idle_timeout',
            attempt
          });
          return;
        }

        if (attempt.abortReason === 'timeout_absolute') {
          attempt.status = 'aborted';
          attempt.aborted = true;
          updateAttemptsDebug();

          resolve({
            type: 'absolute_timeout',
            attempt
          });
          return;
        }

        if (isAbortLikeError(error)) {
          attempt.status = 'aborted';
          attempt.aborted = true;
          attempt.abortReason = attempt.abortReason || 'aborted';
          updateAttemptsDebug();

          resolve({
            type: attempt.abortReason === 'client_closed' ? 'client_closed' : 'aborted',
            attempt
          });
          return;
        }

        attempt.status = 'stream_error';
        attempt.error = error;
        updateAttemptsDebug();

        resolve({
          type: 'stream_error',
          attempt,
          error
        });
      };

      stream.on('data', onData);
      stream.on('end', onEnd);
      stream.on('error', onError);

      resetIdleTimer();
      processChunk(firstChunk);
      stream.resume();
    });
  }

  res.on('close', () => {
    if (res.writableEnded) return;
    markClientClosed();
  });

  try {
    if (!NIM_API_KEY) {
      finalizeRecentRequest(requestId, {
        status: 'configuration_error',
        duration_ms: Date.now() - startedAt,
        error: 'NIM_API_KEY is not configured'
      });

      return res.status(500).json(
        createOpenAIError(500, 'NIM_API_KEY is not configured', 'configuration_error')
      );
    }

    const {
      model,
      messages
    } = body;

    if (!model) {
      finalizeRecentRequest(requestId, {
        status: 'rejected',
        duration_ms: Date.now() - startedAt,
        error: 'Missing required field: model'
      });

      return res.status(400).json(createOpenAIError(400, 'Missing required field: model'));
    }

    if (!Array.isArray(messages)) {
      finalizeRecentRequest(requestId, {
        status: 'rejected',
        duration_ms: Date.now() - startedAt,
        error: 'Missing or invalid required field: messages'
      });

      return res.status(400).json(createOpenAIError(400, 'Missing or invalid required field: messages'));
    }

    const normalizedMessages = normalizeMessages(messages);

    let nimModel = MODEL_MAPPING[model];
    let modelProbeUsed = false;

    // If model is not in map, try it directly
    if (!nimModel) {
      modelProbeUsed = true;

      try {
        const probe = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          {
            model,
            messages: [{ role: 'user', content: 'test' }],
            max_tokens: 1
          },
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },
            validateStatus: (status) => status < 500,
            timeout: 10000
          }
        );

        if (probe.status >= 200 && probe.status < 300) {
          nimModel = model;
        }
      } catch {
        // ignore and fallback below
      }

      if (!nimModel) {
        const modelLower = String(model).toLowerCase();

        if (
          modelLower.includes('gpt-4') ||
          modelLower.includes('claude-opus') ||
          modelLower.includes('405b')
        ) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (
          modelLower.includes('claude') ||
          modelLower.includes('gemini') ||
          modelLower.includes('70b')
        ) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    updateRecentRequest(requestId, {
      status: 'upstream_pending',
      nim_model: nimModel,
      model_probe_used: modelProbeUsed
    });

    const maxTokens =
      typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : 64000;

    const forwardedOptions = pickDefined(body, NIM_FORWARD_OPTION_KEYS);

    // ВСЕГДА stream=true на upstream к NIM
    const nimRequest = {
      model: nimModel,
      messages: normalizedMessages,
      stream: true,
      max_tokens: maxTokens,
      ...forwardedOptions
    };

    if (nimRequest.temperature === undefined) {
      nimRequest.temperature = 0.7;
    }

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = {
        chat_template_kwargs: {
          thinking: true
        }
      };
    }

    updateRecentRequest(requestId, {
      request_options_forwarded: {
        stream: nimRequest.stream,
        max_tokens: nimRequest.max_tokens,
        ...sanitizeDebugObject(nimRequest, NIM_FORWARD_OPTION_KEYS),
        extra_body: nimRequest.extra_body ? nimRequest.extra_body : undefined
      }
    });

    const maxAttempts = NIM_TIMEOUT_RETRY_SWITCH
      ? Math.max(1, NIM_TIMEOUT_MAX_RETRIES)
      : 1;

    for (let attemptIndex = 1; attemptIndex <= maxAttempts; attemptIndex += 1) {
      if (state.clientClosed || state.finalized) {
        return;
      }

      if (remainingAbsoluteMs() <= 0) {
        finalizeWithError(
          model,
          nimModel,
          504,
          `Upstream absolute timeout after ${NIM_TIMEOUT_MS} ms`,
          'timeout',
          'absolute'
        );
        return;
      }

      const firstStageResult = await runAttemptUntilFirstChunk(attemptIndex, nimRequest);

      if (state.clientClosed || state.finalized) {
        return;
      }

      if (firstStageResult.type === 'first_chunk') {
        const winnerResult = await consumeWinningStream(
          model,
          nimModel,
          firstStageResult.attempt,
          firstStageResult.response,
          firstStageResult.firstChunk
        );

        if (state.clientClosed || state.finalized) {
          return;
        }

        if (winnerResult.type === 'completed') {
          finalizeSuccess(model, nimModel);
          return;
        }

        if (winnerResult.type === 'stream_idle_timeout') {
          finalizeWithError(
            model,
            nimModel,
            504,
            `Stream idle timeout after ${NIM_STREAM_IDLE_TIMEOUT_MS} ms`,
            'timeout',
            'stream_idle'
          );
          return;
        }

        if (winnerResult.type === 'absolute_timeout') {
          finalizeWithError(
            model,
            nimModel,
            504,
            `Upstream absolute timeout after ${NIM_TIMEOUT_MS} ms`,
            'timeout',
            'absolute'
          );
          return;
        }

        if (winnerResult.type === 'stream_error') {
          const error = winnerResult.error;
          const status = error?.response?.status || 502;
          const message =
            error?.response?.data?.error?.message ||
            (typeof error?.response?.data === 'string' ? error.response.data : error?.message) ||
            'Upstream stream failed';

          finalizeWithError(
            model,
            nimModel,
            status,
            message,
            'stream_error',
            null
          );
          return;
        }

        finalizeWithError(
          model,
          nimModel,
          502,
          'Winner stream ended unexpectedly',
          'stream_error',
          null
        );
        return;
      }

      if (firstStageResult.type === 'timeout') {
        updateRecentRequest(requestId, {
          timed_out_stage: firstStageResult.stage
        });

        if (firstStageResult.retryable && attemptIndex < maxAttempts && NIM_TIMEOUT_RETRY_SWITCH) {
          updateRecentRequest(requestId, {
            status: 'retrying',
            timed_out_stage: firstStageResult.stage
          });

          if (NIM_TIMEOUT_RETRY_DELAY_MS > 0) {
            const delayMs = Math.min(
              NIM_TIMEOUT_RETRY_DELAY_MS,
              Math.max(0, remainingAbsoluteMs())
            );

            if (delayMs > 0) {
              await sleep(delayMs);
            }
          }

          continue;
        }

        finalizeWithError(
          model,
          nimModel,
          504,
          firstStageResult.message,
          'timeout',
          firstStageResult.stage
        );
        return;
      }

      if (firstStageResult.type === 'ended_before_first_chunk') {
        finalizeWithError(
          model,
          nimModel,
          502,
          firstStageResult.message || 'Upstream stream ended before first chunk',
          'error',
          null
        );
        return;
      }

      if (firstStageResult.type === 'http_error') {
        const error = firstStageResult.error;
        let message = 'Upstream request failed';

        if (error?.response?.data?.error?.message) {
          message = error.response.data.error.message;
        } else if (typeof error?.response?.data === 'string') {
          message = error.response.data;
        } else if (error?.message) {
          message = error.message;
        }

        finalizeWithError(
          model,
          nimModel,
          error.response.status,
          message,
          'error',
          null
        );
        return;
      }

      if (firstStageResult.type === 'error_before_first_chunk') {
        const error = firstStageResult.error;
        const status = error?.response?.status || 502;
        const message =
          error?.response?.data?.error?.message ||
          (typeof error?.response?.data === 'string' ? error.response.data : error?.message) ||
          'Upstream request failed before first chunk';

        finalizeWithError(
          model,
          nimModel,
          status,
          message,
          'error',
          null
        );
        return;
      }

      if (firstStageResult.type === 'client_closed') {
        return;
      }

      if (firstStageResult.type === 'aborted') {
        finalizeWithError(
          model,
          nimModel,
          502,
          'Upstream request aborted',
          'error',
          null
        );
        return;
      }
    }

    finalizeWithError(
      body.model,
      MODEL_MAPPING[body.model] || body.model || null,
      502,
      'All upstream attempts failed',
      'error',
      null
    );
  } catch (error) {
    console.error(`[${requestId}] Proxy error:`, error.response?.data || error.message);

    let status = error.response?.status || 500;
    let message = error.message || 'Internal server error';
    let errorStatus = 'error';

    if (error.code === 'ECONNABORTED') {
      status = 504;
      message = `Upstream timeout after ${NIM_TIMEOUT_MS} ms`;
      errorStatus = 'timeout';
    } else if (error.response?.data?.error?.message) {
      message = error.response.data.error.message;
    } else if (typeof error.response?.data === 'string') {
      message = error.response.data;
    }

    finalizeRecentRequest(requestId, {
      status: errorStatus,
      duration_ms: Date.now() - startedAt,
      nim_status: error.response?.status || null,
      error: message
    });

    if (!res.headersSent) {
      return res
        .status(status)
        .json(
          createOpenAIError(
            status,
            message,
            errorStatus === 'timeout' ? 'timeout_error' : openAIErrorTypeByStatus(status)
          )
        );
    }

    res.end();
  }
});

// Optional root info
app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to NVIDIA NIM Proxy',
    endpoints: [
      'GET /health',
      'GET /debug/recent-requests',
      'GET /v1/models',
      'POST /v1/chat/completions'
    ]
  });
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json(
    createOpenAIError(404, `Endpoint ${req.path} not found`)
  );
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Debug recent requests: http://localhost:${PORT}/debug/recent-requests`);
  console.log(`NIM API base: ${NIM_API_BASE}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`API key configured: ${NIM_API_KEY ? 'YES' : 'NO'}`);
  console.log(`NIM timeout: ${NIM_TIMEOUT_MS} ms`);
  console.log(`NIM response headers timeout: ${NIM_RESPONSE_HEADERS_TIMEOUT_MS} ms`);
  console.log(`NIM first chunk timeout: ${NIM_FIRST_CHUNK_TIMEOUT_MS} ms`);
  console.log(`NIM stream idle timeout: ${NIM_STREAM_IDLE_TIMEOUT_MS} ms`);
  console.log(`Retry enabled: ${NIM_TIMEOUT_RETRY_SWITCH ? 'YES' : 'NO'}`);
  console.log(`Retry max attempts: ${NIM_TIMEOUT_MAX_RETRIES}`);
  console.log(`Retry delay: ${NIM_TIMEOUT_RETRY_DELAY_MS} ms`);
  console.log(`Slow request threshold: ${SLOW_REQUEST_MS} ms`);
});
