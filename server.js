// server.js - OpenAI to NVIDIA NIM API Proxy (Chub-friendly)
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
  const base = `[${record.id}] ${record.status} model=${record.client_model} -> ${record.nim_model || 'n/a'} duration=${record.duration_ms ?? 'n/a'}ms`;
  if ((record.duration_ms ?? 0) >= SLOW_REQUEST_MS) {
    console.warn('[slow-request]', base);
  } else {
    console.log('[request]', base);
  }
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

  addRecentRequest({
    id: requestId, // internal only, not returned by debug endpoint
    started_at: new Date(startedAt).toISOString(),
    status: 'received',
    client_model: body.model || null,
    nim_model: null,
    stream: Boolean(body.stream),
    message_count: Array.isArray(body.messages) ? body.messages.length : 0,
    body_keys: safeBodyKeys(body),
    requested_max_tokens:
      typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : 64000,
    request_options_received: sanitizeDebugObject(body, DEBUG_RECEIVED_OPTION_KEYS)
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
      messages,
      stream
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
      } catch (e) {
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

    const nimRequest = {
      model: nimModel,
      messages: normalizedMessages,
      stream: Boolean(stream),
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

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: NIM_TIMEOUT_MS
      }
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;
      let firstChunkAt = null;
      let finalized = false;

      response.data.on('data', (chunk) => {
        if (!firstChunkAt) {
          firstChunkAt = Date.now();
          updateRecentRequest(requestId, {
            status: 'streaming',
            nim_status: response.status,
            first_byte_ms: firstChunkAt - startedAt
          });
        }

        buffer += chunk.toString('utf8');
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const rawLine of lines) {
          const line = rawLine.trimEnd();
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));

            if (data.choices?.[0]?.delta) {
              const delta = data.choices[0].delta;

              const reasoning = contentToString(delta.reasoning_content);
              const content = contentToString(delta.content);

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent += '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent += reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += '</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                delta.content = combinedContent || '';
                delete delta.reasoning_content;
              } else {
                delta.content = content || '';
                delete delta.reasoning_content;
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n\n');
          }
        }
      });

      response.data.on('end', () => {
        if (finalized) return;
        finalized = true;

        const durationMs = Date.now() - startedAt;

        finalizeRecentRequest(requestId, {
          status: 'completed',
          duration_ms: durationMs,
          nim_status: response.status,
          first_byte_ms: firstChunkAt ? firstChunkAt - startedAt : null
        });

        logRequestSummary(getRequestLogById(requestId) || {
          id: requestId,
          status: 'completed',
          client_model: model,
          nim_model: nimModel,
          duration_ms: durationMs
        });

        res.end();
      });

      response.data.on('error', (err) => {
        if (finalized) return;
        finalized = true;

        const durationMs = Date.now() - startedAt;

        finalizeRecentRequest(requestId, {
          status: 'stream_error',
          duration_ms: durationMs,
          nim_status: response.status,
          first_byte_ms: firstChunkAt ? firstChunkAt - startedAt : null,
          error: err.message
        });

        console.error(`[${requestId}] Stream error:`, err.message);
        res.end();
      });

      req.on('close', () => {
        if (!finalized && !res.writableEnded) {
          finalized = true;

          finalizeRecentRequest(requestId, {
            status: 'client_closed',
            duration_ms: Date.now() - startedAt,
            nim_status: response.status,
            first_byte_ms: firstChunkAt ? firstChunkAt - startedAt : null
          });
        }

        if (response.data?.destroy) {
          response.data.destroy();
        }
      });

      return;
    }

    const sourceData = response.data || {};

    const openaiResponse = {
      id: sourceData.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: sourceData.created || Math.floor(Date.now() / 1000),
      model,
      choices: (sourceData.choices || []).map((choice) => {
        let fullContent = contentToString(choice?.message?.content);
        const reasoningText = contentToString(choice?.message?.reasoning_content);

        if (SHOW_REASONING && reasoningText) {
          fullContent = `<think>\n${reasoningText}\n</think>\n\n${fullContent}`;
        }

        return {
          index: choice?.index ?? 0,
          message: {
            role: choice?.message?.role || 'assistant',
            content: fullContent
          },
          finish_reason: choice?.finish_reason || 'stop'
        };
      }),
      usage: sourceData.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    const durationMs = Date.now() - startedAt;

    finalizeRecentRequest(requestId, {
      status: 'completed',
      duration_ms: durationMs,
      nim_status: response.status,
      usage: sourceData.usage || null,
      choice_count: Array.isArray(sourceData.choices) ? sourceData.choices.length : 0
    });

    logRequestSummary(getRequestLogById(requestId) || {
      id: requestId,
      status: 'completed',
      client_model: model,
      nim_model: nimModel,
      duration_ms: durationMs
    });

    return res.json(openaiResponse);
  } catch (error) {
    console.error(`[${requestId}] Proxy error:`, error.response?.data || error.message);

    let status = error.response?.status || 500;
    let message = error.message || 'Internal server error';
    let errorStatus = 'error';

    if (error.code === 'ECONNABORTED') {
      status = 504;
      message = `Upstream NIM timeout after ${NIM_TIMEOUT_MS} ms`;
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
      .json(createOpenAIError(status, message));
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
  console.log(`Slow request threshold: ${SLOW_REQUEST_MS} ms`);
});
