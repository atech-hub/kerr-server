# kerr-server

OpenAI-compatible API server for [Kerr-ODE](https://github.com/atech-hub/kerr-engine) models. Load a trained checkpoint, serve it via HTTP. Any chat UI that speaks the OpenAI protocol connects without modification.

## Quick Start

```bash
# Build
cargo build --release

# Serve a model (v2 checkpoint — self-describing, no config flags needed)
kerr-server checkpoint.bin data/input.txt --port 8080

# Test
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Usage

```
kerr-server <checkpoint> <data> [options]

Arguments:
  <checkpoint>      Path to checkpoint file (.bin)
  <data>            Path to training data (for vocabulary)

Options:
  --port N          Listen port (default: 8080)
  --host ADDR       Bind address (default: 127.0.0.1)
  --model-name S    Model name in API responses (default: kerr-ode)
  --word            Use word-level tokenizer

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

**POST /v1/chat/completions** — OpenAI-compatible chat completion

Request fields: `messages` (required), `temperature`, `top_p`, `max_tokens`, `stream`, `top_k`, `repetition_penalty`

Non-streaming returns JSON. Streaming (`"stream": true`) returns Server-Sent Events.

**GET /v1/models** — List available models

**GET /health** — Server health check with model info

## Connecting Chat UIs

**LM Studio:** Settings → Override Base URL → `http://127.0.0.1:8080/v1`. Select any model from dropdown — the server uses its loaded model regardless of name.

**Open WebUI / SillyTavern / continue.dev:** Point the OpenAI API base URL to `http://127.0.0.1:8080/v1`. Put any string in the API key field.

## Architecture

~640 lines across 6 modules:

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
- BPE tokenizer support (needed for hybrid models using Qwen/Llama tokenizers)
- GPU backend selection for inference (currently CPU only)
- Model hot-reload without server restart

---

## License

Apache 2.0

## Credits

- **Marco Da Cunha (atech-hub):** Architecture, direction
- **Claude (Anthropic):** Implementation

Part of the [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) project.
