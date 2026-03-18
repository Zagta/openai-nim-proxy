// server.js - OpenAI to NVIDIA NIM API Proxy (Chub-friendly)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '4mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Show/hide reasoning in output
const SHOW_REASONING = false;

// DeepSeek-V3.2 thinking mode is left disabled by default
const ENABLE_THINKING_MODE = true;

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

    // If Chub or another client sends null/empty content, keep it as empty string
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

function createOpenAIError(status, message, type = 'invalid_request_error') {
  return {
    error: {
      message,
      type,
      code: status
    }
  };
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    nim_api_base: NIM_API_BASE,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    api_key_configured: Boolean(NIM_API_KEY)
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
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json(
        createOpenAIError(500, 'NIM_API_KEY is not configured', 'configuration_error')
      );
    }

    const body = req.body || {};
    const {
      model,
      messages,
      stream
    } = body;

    if (!model) {
      return res.status(400).json(createOpenAIError(400, 'Missing required field: model'));
    }

    if (!Array.isArray(messages)) {
      return res.status(400).json(createOpenAIError(400, 'Missing or invalid required field: messages'));
    }

    const normalizedMessages = normalizeMessages(messages);

    let nimModel = MODEL_MAPPING[model];

    // If model is not in map, try it directly
    if (!nimModel) {
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
            validateStatus: (status) => status < 500
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

    const maxTokens =
      typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : 4096;

    const nimRequest = {
      model: nimModel,
      messages: normalizedMessages,
      stream: Boolean(stream),
      max_tokens: maxTokens,
      ...pickDefined(body, [
        'temperature',
        'top_p',
        'presence_penalty',
        'frequency_penalty',
        'stop',
        'seed',
        'n',
        'user'
      ])
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

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json'
      }
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
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
            // Pass through raw line if parse fails
            res.write(line + '\n\n');
          }
        }
      });

      response.data.on('end', () => {
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

      req.on('close', () => {
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

    return res.json(openaiResponse);
  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);

    let message = error.message || 'Internal server error';

    if (error.response?.data?.error?.message) {
      message = error.response.data.error.message;
    } else if (typeof error.response?.data === 'string') {
      message = error.response.data;
    }

    return res
      .status(error.response?.status || 500)
      .json(createOpenAIError(error.response?.status || 500, message));
  }
});

// Optional root info
app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to NVIDIA NIM Proxy',
    endpoints: [
      'GET /health',
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
  console.log(`NIM API base: ${NIM_API_BASE}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`API key configured: ${NIM_API_KEY ? 'YES' : 'NO'}`);
});
