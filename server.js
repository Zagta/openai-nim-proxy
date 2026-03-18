// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Shows/hides reasoning in output
const SHOW_REASONING = false;

// Enables thinking for models that support chat_template_kwargs.thinking
// Для DeepSeek-V3.2 это не гарантированно подтверждено документацией NVIDIA,
// поэтому по умолчанию выключено.
const ENABLE_THINKING_MODE = false;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',

  // Можешь использовать как "gpt-4o", так и напрямую "deepseek-v3.2"
  'gpt-4o': 'deepseek-ai/deepseek-v3.2',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2',

  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Normalize OpenAI-style roles to what NVIDIA endpoint expects
function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return [];

  return messages.map((msg) => {
    const role = msg?.role === 'developer' ? 'system' : msg?.role;

    return {
      ...msg,
      role
    };
  });
}

// Health check endpoint
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

// List models endpoint (OpenAI compatible)
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

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'NIM_API_KEY is not configured',
          type: 'configuration_error',
          code: 500
        }
      });
    }

    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body || {};

    if (!model) {
      return res.status(400).json({
        error: {
          message: 'Missing required field: model',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    if (!Array.isArray(messages)) {
      return res.status(400).json({
        error: {
          message: 'Missing or invalid required field: messages',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    const normalizedMessages = normalizeMessages(messages);

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      try {
        const probeResponse = await axios.post(
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

        if (probeResponse.status >= 200 && probeResponse.status < 300) {
          nimModel = model;
        }
      } catch (e) {
        // ignore probe errors, fallback below
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

    const nimRequest = {
      model: nimModel,
      messages: normalizedMessages,
      temperature: typeof temperature === 'number' ? temperature : 0.6,
      max_tokens: typeof max_tokens === 'number' ? max_tokens : 9024,
      stream: Boolean(stream)
    };

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
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));

            if (data.choices?.[0]?.delta) {
              const delta = data.choices[0].delta;
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += '</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                if (combinedContent) {
                  data.choices[0].delta.content = combinedContent;
                } else {
                  data.choices[0].delta.content = '';
                }

                delete data.choices[0].delta.reasoning_content;
              } else {
                data.choices[0].delta.content = content || '';
                delete data.choices[0].delta.reasoning_content;
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
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

      return;
    }

    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: (response.data.choices || []).map((choice) => {
        let fullContent = choice.message?.content || '';

        if (SHOW_REASONING && choice.message?.reasoning_content) {
          fullContent =
            '<think>\n' +
            choice.message.reasoning_content +
            '\n</think>\n\n' +
            fullContent;
        }

        return {
          index: choice.index ?? 0,
          message: {
            role: choice.message?.role || 'assistant',
            content: fullContent
          },
          finish_reason: choice.finish_reason || 'stop'
        };
      }),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);
  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);

    let message = error.message || 'Internal server error';

    if (error.response?.data?.error?.message) {
      message = error.response.data.error.message;
    } else if (typeof error.response?.data === 'string') {
      message = error.response.data;
    }

    res.status(error.response?.status || 500).json({
      error: {
        message,
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`NIM API base: ${NIM_API_BASE}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`API key configured: ${NIM_API_KEY ? 'YES' : 'NO'}`);
});
