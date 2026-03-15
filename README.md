# kerr-server

OpenAI-compatible API server for [Kerr-ODE](https://github.com/atech-hub/kerr-engine) models. Load a trained checkpoint, serve it via HTTP. Any chat UI that speaks the OpenAI protocol connects without modification.

## Quick Start

```bash
# Build
cargo build --release

# See all options
kerr-server --help

# Serve a model (v2 checkpoint — self-describing, no config flags needed)
kerr-server checkpoint.bin data/input.txt --port 8080

# Serve with BPE tokenizer (no data file needed)
kerr-server checkpoint.bin --bpe tokenizer.json --port 8080

# Serve with API key authentication
kerr-server checkpoint.bin data/input.txt --port 8080 --api-key sk-your-secret-key

# Test
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-secret-key" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Usage

```
kerr-server <checkpoint> [data] [options]

Arguments:
  <checkpoint>      Path to checkpoint file (.bin)
  [data]            Path to training data (for vocabulary — optional with --bpe)

Tokenizer (one of):
  [data]            Character/word vocab extracted from training data
  --bpe FILE        BPE tokenizer from HuggingFace tokenizer.json (Qwen, Llama, GPT-2)
  --word            Use word-level tokenizer (with data file)

Server:
  --port N          Listen port (default: 8080)
  --host ADDR       Bind address (default: 127.0.0.1)
  --model-name S    Model name in API responses (default: kerr-ode)
  --api-key KEY     Require bearer token auth on /v1/* endpoints (no flag = no auth)

Architecture (v1 checkpoints only — v2 self-describes):
  --n-bands N       Harmonic frequency bands (default: 64)
  --n-head N        Attention heads (default: 4)
  --n-layers N      Transformer blocks (default: 4)
  --maestro-dim N   Maestro bottleneck width (default: 16)
  --block-size N    Max sequence length (default: 256)
  --rk4-steps N     ODE integration steps (default: 8)
```

## Checkpoint Versions

**v2 (recommended):** Self-describing. The checkpoint header contains the full ModelConfig. No architecture flags needed — just point at the file.

```bash
kerr-server checkpoint.bin data/input.txt --port 8080
```

**v1 (legacy):** Requires architecture flags if the model isn't the default 128-dim config.

```bash
kerr-server checkpoint_768.bin data/input.txt --n-bands 384 --n-head 12 --maestro-dim 48
```

Checkpoints saved by kerr-engine v0.2.0+ use v2 format. Older v1 checkpoints are fully supported.

## Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/v1/chat/completions` | POST | Yes | OpenAI-compatible chat completion |
| `/v1/models` | GET | Yes | List available models |
| `/health` | GET | No | Server health check with model info |

**POST /v1/chat/completions** request fields: `messages` (required), `temperature`, `top_p`, `max_tokens`, `stream`, `top_k`, `repetition_penalty`.

Non-streaming returns JSON. Streaming (`"stream": true`) returns Server-Sent Events with `data: {chunk}\n\n` format, ending with `data: [DONE]\n\n`.

**Auth:** When `--api-key` is set, all `/v1/*` endpoints require `Authorization: Bearer <key>` header. `/health` is always open (for load balancers and monitoring).

## Connecting Chat UIs

Any chat application that supports OpenAI-compatible endpoints can connect to the Kerr Server.

**LM Studio (verified):** Install the `openai-compat-endpoint` plugin from the LM Studio plugin store. In the chat panel, set "Override Base URL" to `http://127.0.0.1:8080/v1`. Select any model from the dropdown — the server uses its loaded model regardless of the name sent in the request. Put your API key in the "OpenAI API Key" field (or any string if auth is disabled). Tested and verified with LM Studio 0.4.6 — a 354K parameter Shakespeare model responded through the GPT-4.1 label.

**Open WebUI:** Settings → Connections → add `http://127.0.0.1:8080/v1` as an OpenAI endpoint.

**SillyTavern / continue.dev / any OpenAI client:** Point the API base URL to `http://127.0.0.1:8080/v1`. Set the API key if auth is enabled.

## Architecture

~750 lines across 6 modules:

- `main.rs` — CLI parsing, checkpoint + vocab loading, server startup
- `server.rs` — Axum router, shared state, graceful shutdown (Ctrl+C)
- `api_types.rs` — OpenAI protocol types (pure serde structs)
- `inference.rs` — Token generation with temperature/top-k/top-p/repetition penalty sampling
- `prompt.rs` — Vocabulary extraction, text encode/decode, chat message formatting
- `handlers.rs` — Request handlers, SSE streaming

Uses `model.forward()` (CPU inference path). No KV-cache — full context re-run per token. Fast at 128-dim.

## Dependencies

- [kerr-engine](https://github.com/atech-hub/kerr-engine) — model weights, checkpoint loader, tokenizer
- [axum](https://github.com/tokio-rs/axum) 0.8 — HTTP framework
- [tokio](https://tokio.rs) — async runtime
- serde, serde_json, uuid, tokio-stream

## Requirements

- Rust nightly-2025-11-13 (matches kerr-engine toolchain)
- kerr-engine repo at `../kerr-engine` (path dependency)

## Contributing

The maintainer (Marco Da Cunha) is an IT systems administrator, not a programmer. The server was built through collaboration with AI (Claude Desktop for architecture, Claude Code for implementation). This is stated openly.

What this means for contributions:

- **Main branch is protected.** All changes go through pull requests.
- **Fork and branch.** Want to add KV-cache, improve sampling, add new endpoints? Fork the repo, create a branch, do your work, submit a PR.
- **The validation gate is the review.** Every PR must demonstrate that the four endpoints still work correctly — health, models, non-streaming chat, and SSE streaming.
- **The maintainer merges based on testing and description, not code review.** Be clear about what you changed and why.

**Known targets for contributors:**
- KV-cache for efficient generation at 768-dim+ (essential for production use)
- Vocab embedded in checkpoint (eliminate the data file requirement at serve time)
- GPU backend selection for inference (currently CPU only)
- Model hot-reload without server restart
- Concurrent request handling (currently sequential inference)

---

## Related

- [Kerr Engine](https://github.com/atech-hub/kerr-engine) — Training engine that produces the checkpoints this server serves (public, Apache 2.0)
- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project (public, MIT)

---

## License

Apache 2.0. See [LICENSE](LICENSE).

## Credits

- **Marco Da Cunha** — Architecture, direction, pattern recognition
- **Claude Desktop (Opus)** — Architecture design, documentation
- **Claude Code** — Implementation, testing

Part of the [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) project.
