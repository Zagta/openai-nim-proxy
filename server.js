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

// Timeouts / logging config
const NIM_TIMEOUT_MS = Number(process.env.NIM_TIMEOUT_MS || 180000);
const NIM_RESPONSE_HEADERS_TIMEOUT_MS = Number(process.env.NIM_RESPONSE_HEADERS_TIMEOUT_MS || 15000);
const NIM_FIRST_CHUNK_TIMEOUT_MS = Number(process.env.NIM_FIRST_CHUNK_TIMEOUT_MS || 30000);
const NIM_STREAM_IDLE_TIMEOUT_MS = Number(process.env.NIM_STREAM_IDLE_TIMEOUT_MS || 20000);
const NIM_TIMEOUT_HEDGE_SWITCH = /^(1|true|yes|on)$/i.test(String(process.env.NIM_TIMEOUT_HEDGE_SWITCH || 'false'));
const NIM_HEDGE_DELAY_MS = Number(process.env.NIM_HEDGE_DELAY_MS || 8000);

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
    nim_timeout_hedge_switch: NIM_TIMEOUT_HEDGE_SWITCH,
    nim_hedge_delay_ms: NIM_HEDGE_DELAY_MS,
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
    hedge_enabled: Boolean(r.hedge_enabled),
    hedge_delay_ms: r.hedge_delay_ms ?? null,
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
    hedge_enabled: NIM_TIMEOUT_HEDGE_SWITCH,
    hedge_delay_ms: NIM_TIMEOUT_HEDGE_SWITCH ? NIM_HEDGE_DELAY_MS : null,
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

    // Вариант A: ВСЕГДА stream=true на upstream к NIM
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

    const maxAttempts = NIM_TIMEOUT_HEDGE_SWITCH ? 2 : 1;
    const state = {
      finalized: false,
      clientStreamRequested: downstreamStreamRequested,
      clientHeadersSent: false,
      winnerAttempt: null,
      attempts: [],
      bestError: null,
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

    let absoluteTimer = null;
    let firstChunkTimer = null;
    let idleTimer = null;
    let hedgeTimer = null;

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

    function clearAllTimers() {
      clearTimeout(absoluteTimer);
      clearTimeout(firstChunkTimer);
      clearTimeout(idleTimer);
      clearTimeout(hedgeTimer);
    }

    function resetIdleTimer() {
      clearTimeout(idleTimer);

      idleTimer = setTimeout(() => {
        finalizeWithError(
          504,
          `Stream idle timeout after ${NIM_STREAM_IDLE_TIMEOUT_MS} ms`,
          'timeout',
          'stream_idle'
        );
      }, NIM_STREAM_IDLE_TIMEOUT_MS);
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

    function abortAttempt(attempt, reason = 'aborted') {
      if (!attempt || attempt.aborted) return;

      attempt.aborted = true;
      attempt.status = 'aborted';
      attempt.abortReason = reason;

      try {
        attempt.controller.abort();
      } catch {
        // ignore
      }

      safeDestroyStream(attempt.response?.data);
      updateAttemptsDebug();
    }

    function abortAllAttempts(reason = 'aborted') {
      for (const attempt of state.attempts) {
        abortAttempt(attempt, reason);
      }
    }

    function abortLosers(winnerIndex) {
      for (const attempt of state.attempts) {
        if (attempt.index !== winnerIndex) {
          abortAttempt(attempt, 'hedge_loser');
        }
      }
    }

    function buildOpenAIResponseFromAggregate() {
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

    function finalizeSuccess() {
      if (state.finalized) return;
      state.finalized = true;
      clearAllTimers();

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

      return res.json(buildOpenAIResponseFromAggregate());
    }

    function finalizeWithError(status, message, errorStatus = 'error', timedOutStage = null) {
      if (state.finalized) return;
      state.finalized = true;
      clearAllTimers();

      const durationMs = Date.now() - startedAt;
      const winner = state.attempts.find((a) => a.index === state.winnerAttempt) || null;

      abortAllAttempts(errorStatus === 'timeout' ? 'timeout' : 'error');

      finalizeRecentRequest(requestId, {
        status: errorStatus,
        duration_ms: durationMs,
        nim_status: winner?.nimStatus ?? state.bestError?.response?.status ?? null,
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

    function maybeFailEarlyIfAllAttemptsDead() {
      if (state.finalized || state.winnerAttempt) return;

      const canStartMore = state.attempts.length < maxAttempts;
      const anyActive = state.attempts.some((attempt) => !attempt.settled && !attempt.aborted);

      if (canStartMore || anyActive) {
        return;
      }

      if (state.bestError?.code === 'ECONNABORTED') {
        finalizeWithError(
          504,
          `Upstream response headers timeout after ${NIM_RESPONSE_HEADERS_TIMEOUT_MS} ms`,
          'timeout',
          'response_headers'
        );
        return;
      }

      if (state.bestError?.response?.status) {
        let message = 'Upstream request failed';
        if (state.bestError.response?.data?.error?.message) {
          message = state.bestError.response.data.error.message;
        } else if (typeof state.bestError.response?.data === 'string') {
          message = state.bestError.response.data;
        } else if (state.bestError.message) {
          message = state.bestError.message;
        }

        finalizeWithError(
          state.bestError.response.status,
          message,
          'error',
          'response_headers'
        );
        return;
      }

      finalizeWithError(
        502,
        state.bestError?.message || 'All upstream attempts failed',
        isAbortLikeError(state.bestError) ? 'timeout' : 'error',
        state.bestError?.code === 'ECONNABORTED' ? 'response_headers' : null
      );
    }

    function maybeStartHedge() {
      if (!NIM_TIMEOUT_HEDGE_SWITCH) return;
      if (state.finalized || state.winnerAttempt) return;
      if (state.attempts.length >= maxAttempts) return;
      startAttempt(state.attempts.length + 1);
    }

    function onWinnerSelected(attempt) {
      if (state.winnerAttempt) return;
      state.winnerAttempt = attempt.index;

      clearTimeout(firstChunkTimer);
      clearTimeout(hedgeTimer);

      updateRecentRequest(requestId, {
        status: 'streaming',
        nim_status: attempt.nimStatus,
        first_byte_ms: attempt.firstDataAt - startedAt,
        winner_attempt: attempt.index
      });

      abortLosers(attempt.index);
      updateAttemptsDebug();
      resetIdleTimer();
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

    function onAttemptData(attempt, chunk) {
      if (state.finalized) return;
      if (attempt.aborted) return;

      if (!attempt.firstDataAt) {
        attempt.firstDataAt = Date.now();
        attempt.status = 'streaming';
        onWinnerSelected(attempt);
      } else if (state.winnerAttempt === attempt.index) {
        resetIdleTimer();
      }

      if (state.winnerAttempt !== attempt.index) {
        return;
      }

      attempt.buffer += chunk.toString('utf8');
      const lines = attempt.buffer.split('\n');
      attempt.buffer = lines.pop() || '';

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        processWinnerLine(line);
      }

      updateAttemptsDebug();
    }

    function onAttemptEnd(attempt) {
      attempt.settled = true;
      attempt.status = 'completed';
      updateAttemptsDebug();

      if (state.finalized) return;

      if (state.winnerAttempt === attempt.index) {
        finalizeSuccess();
      } else {
        maybeFailEarlyIfAllAttemptsDead();
      }
    }

    function onAttemptFailure(attempt, error, phase = 'error') {
      if (attempt.settled) return;

      attempt.settled = true;

      if (attempt.aborted || isAbortLikeError(error)) {
        attempt.status = 'aborted';
        updateAttemptsDebug();

        if (!state.winnerAttempt) {
          maybeFailEarlyIfAllAttemptsDead();
        }

        return;
      }

      attempt.status = phase === 'stream' ? 'stream_error' : 'error';
      attempt.error = error;

      if (!state.bestError) {
        state.bestError = error;
      }

      updateAttemptsDebug();

      if (!state.winnerAttempt) {
        if (NIM_TIMEOUT_HEDGE_SWITCH && state.attempts.length < maxAttempts) {
          maybeStartHedge();
        }

        maybeFailEarlyIfAllAttemptsDead();
        return;
      }

      if (state.winnerAttempt === attempt.index) {
        const status = error?.code === 'ECONNABORTED' ? 504 : (error?.response?.status || 502);
        const message =
          error?.code === 'ECONNABORTED'
            ? `Stream timeout after ${NIM_STREAM_IDLE_TIMEOUT_MS} ms`
            : error?.response?.data?.error?.message ||
              (typeof error?.response?.data === 'string' ? error.response.data : error?.message) ||
              'Upstream stream failed';

        finalizeWithError(
          status,
          message,
          error?.code === 'ECONNABORTED' ? 'timeout' : 'stream_error',
          error?.code === 'ECONNABORTED' ? 'stream_idle' : null
        );
      }
    }

    function startAttempt(index) {
      if (state.finalized) return;
      if (state.attempts.some((a) => a.index === index)) return;

      const controller = new AbortController();
      const attempt = {
        index,
        controller,
        response: null,
        buffer: '',
        startedAt: Date.now(),
        headersAt: null,
        firstDataAt: null,
        nimStatus: null,
        settled: false,
        aborted: false,
        status: 'starting',
        error: null
      };

      state.attempts.push(attempt);
      updateAttemptsDebug();

      axios.post(
        `${NIM_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: 'stream',
          timeout: NIM_RESPONSE_HEADERS_TIMEOUT_MS,
          signal: controller.signal
        }
      ).then((response) => {
        if (state.finalized || attempt.aborted) {
          safeDestroyStream(response.data);
          return;
        }

        attempt.response = response;
        attempt.nimStatus = response.status;
        attempt.headersAt = Date.now();
        attempt.status = 'headers_received';
        updateAttemptsDebug();

        response.data.on('data', (chunk) => onAttemptData(attempt, chunk));
        response.data.on('end', () => onAttemptEnd(attempt));
        response.data.on('error', (err) => onAttemptFailure(attempt, err, 'stream'));
      }).catch((error) => {
        onAttemptFailure(attempt, error, 'setup');
      });
    }

    // Global client-close handling
    req.on('close', () => {
      if (state.finalized) return;

      state.finalized = true;
      clearAllTimers();
      abortAllAttempts('client_closed');

      finalizeRecentRequest(requestId, {
        status: 'client_closed',
        duration_ms: Date.now() - startedAt,
        attempt_count: state.attempts.length,
        winner_attempt: state.winnerAttempt
      });
    });

    // Global timeouts
    absoluteTimer = setTimeout(() => {
      finalizeWithError(
        504,
        `Upstream absolute timeout after ${NIM_TIMEOUT_MS} ms`,
        'timeout',
        'absolute'
      );
    }, NIM_TIMEOUT_MS);

    firstChunkTimer = setTimeout(() => {
      finalizeWithError(
        504,
        `No first stream chunk after ${NIM_FIRST_CHUNK_TIMEOUT_MS} ms`,
        'timeout',
        'before_first_chunk'
      );
    }, NIM_FIRST_CHUNK_TIMEOUT_MS);

    if (NIM_TIMEOUT_HEDGE_SWITCH) {
      hedgeTimer = setTimeout(() => {
        if (!state.finalized && !state.winnerAttempt) {
          maybeStartHedge();
        }
      }, NIM_HEDGE_DELAY_MS);
    }

    // Start first upstream attempt
    startAttempt(1);

    return;
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

    return res
      .status(status)
      .json(createOpenAIError(status, message, errorStatus === 'timeout' ? 'timeout_error' : openAIErrorTypeByStatus(status)));
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
  console.log(`Hedge enabled: ${NIM_TIMEOUT_HEDGE_SWITCH ? 'YES' : 'NO'}`);
  console.log(`Hedge delay: ${NIM_HEDGE_DELAY_MS} ms`);
  console.log(`Slow request threshold: ${SLOW_REQUEST_MS} ms`);
});
