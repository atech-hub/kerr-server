# kerr-server

OpenAI-compatible API server for [Kerr-ODE](https://github.com/atech-hub/kerr-engine) models. Self-contained — no engine dependency, no GPU code. Load a trained checkpoint, serve it via HTTP, optionally accumulate wave memory across conversations. Any chat UI that speaks the OpenAI protocol connects without modification.

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

# Serve with wave memory (accumulates experience across conversations)
kerr-server checkpoint.bin data/input.txt --memory memory.kwmf --port 8080

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

Wave Memory:
  --memory FILE     Load/create a .kwmf wave memory file. Memory offsets inject into
                    Kerr-ODE initial conditions during inference. Accumulates experience
                    across conversations and saves after each one. Omit for no memory.

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

## Wave Memory

The `--memory` flag enables persistent experience accumulation across conversations. The model weights never change — a separate file (typically 1.5KB) shifts the Kerr-ODE's starting position on the unit circle.

```bash
# First conversation — creates fresh memory file
kerr-server checkpoint.bin data/input.txt --memory memory.kwmf

# Subsequent conversations — loads and accumulates
kerr-server checkpoint.bin data/input.txt --memory memory.kwmf

# Inspect what accumulated
kerr-memory census memory.kwmf
```

How it works: during inference, memory offsets add to the ODE initial conditions (`Z_k = input_k + α · memory_k`). After each conversation, the ODE final states feed an exponential moving average that merges into the persistent file. The Kerr dynamics do the rest — self-phase modulation amplifies resonant memories, cross-phase coupling associates related bands, damping forgets what's not reinforced.

Validated findings: stochastic resonance (α=0.05 improves perplexity by 8.8%), stable accumulation over 20 conversations, bit-identical reset on deletion, anomaly detection catches spikes before affecting output. Word-level model shows semantic tone influence — love memory produces "fair", "give thee" while war memory produces "dishonour", "death" from the same prompt.

See [kerr-memory](https://github.com/atech-hub/kerr-memory) for the library, CLI tools, and full investigation results.

---

## Architecture

~1,900 lines across 10 modules. Self-contained — the forward pass is built in, no kerr-engine dependency.

- `model.rs` — Forward pass, weight structs, Kerr-ODE with memory injection
- `checkpoint.rs` — Checkpoint loader (v1 and v2 formats)
- `data.rs` — Character and word tokenizers
- `bpe.rs` — BPE tokenizer (HuggingFace tokenizer.json)
- `rng.rs` — Deterministic PRNG for sampling
- `server.rs` — Axum router, shared state, graceful shutdown (Ctrl+C)
- `api_types.rs` — OpenAI protocol types (pure serde structs)
- `inference.rs` — Token generation with temperature/top-k/top-p/repetition penalty sampling
- `prompt.rs` — Vocabulary extraction, text encode/decode, chat message formatting
- `handlers.rs` — Request handlers, SSE streaming, memory accumulation

Uses `model.forward()` / `forward_with_memory()` (CPU inference path). No KV-cache — full context re-run per token. Fast at 128-dim.

**Scale limitation:** The server runs CPU-only inference. At 128-dim this is instant (~1ms per token). At 768-dim this is ~1.7s per token — a 100-token response takes nearly 3 minutes, which is impractical for chat. GPU inference is not implemented. If you need to serve models larger than 128-dim, the server needs GPU inference support (contributor target).

## Dependencies

- [kerr-memory](https://github.com/atech-hub/kerr-memory) — wave memory state management (optional, for `--memory` flag)
- [axum](https://github.com/tokio-rs/axum) 0.8 — HTTP framework
- [tokio](https://tokio.rs) — async runtime
- serde, serde_json, uuid, tokio-stream

No kerr-engine dependency. The forward pass (model.rs, checkpoint.rs, data.rs, bpe.rs, rng.rs) is built into the server. No GPU code, no wgpu, no shaders.

## Requirements

- Rust nightly-2025-11-13
- kerr-memory repo at `../kerr-memory` (path dependency, for `--memory` support)

## Contributing

The maintainer (Marco Da Cunha) is an IT systems administrator, not a programmer. The server was built through collaboration with AI (Claude Desktop for architecture, Claude Code for implementation). This is stated openly.

What this means for contributions:

- **Main branch is protected.** All changes go through pull requests.
- **Fork and branch.** Want to add KV-cache, improve sampling, add new endpoints? Fork the repo, create a branch, do your work, submit a PR.
- **The validation gate is the review.** Every PR must demonstrate that the four endpoints still work correctly — health, models, non-streaming chat, and SSE streaming.
- **The maintainer merges based on testing and description, not code review.** Be clear about what you changed and why.

**Known targets for contributors (priority order):**
- **GPU inference** (critical) — the server is CPU-only. At 768-dim, inference is ~1.7s per token, making chat impractical. This is the #1 blocker for serving larger models.
- **KV-cache** (critical at scale) — without it, the full context is recomputed per token. At 128-dim this doesn't matter. At 768-dim+ it's the difference between usable and unusable.
- Vocab embedded in checkpoint (eliminate the data file requirement at serve time)
- Streaming memory accumulation (currently only non-streaming requests update memory)
- Model hot-reload without server restart
- Concurrent request handling (currently sequential inference)

---

## Related

- [Kerr Engine](https://github.com/atech-hub/kerr-engine) — Training engine that produces the checkpoints this server serves (public, Apache 2.0)
- [Kerr Memory](https://github.com/atech-hub/kerr-memory) — Wave memory library used by the `--memory` flag (public, Apache 2.0)
- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project (public, MIT)

---

## License

Apache 2.0. See [LICENSE](LICENSE).

## Credits

- **Marco Da Cunha** — Architecture, direction, pattern recognition
- **Claude Desktop (Opus)** — Architecture design, documentation
- **Claude Code** — Implementation, testing

Part of the [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) project.
