# kerr-server

> **⚠️ This repository is the original prototype server and is no longer under active development.**
>
> The project has evolved into a new architecture. The production inference server is now **[wave-server](https://github.com/atech-hub/wave-server)**, which features:
>
> - **Parallel block architecture** (GPT-J formulation) matching the wave-engine training architecture exactly
> - **KV-cache** for harmonic coherence attention — cached phase angles and value projections make generation dramatically faster at larger models
> - **Dual-maestro Kerr-ODE** — pre-ODE and post-ODE coordination matching wave-engine's training architecture
> - **WCHK checkpoint format** — self-describing headers with full model config, plus backward-compatible KCHK loading
> - **GPU acceleration** (optional) — wgpu matmul dispatch for out_proj and lm_head
>
> The kerr-server remains available as a historical reference and continues to work with kerr-engine checkpoints. It will not receive new features.
>
> **New repos:**
> - **[wave-server](https://github.com/atech-hub/wave-server)** — Production inference server (Apache 2.0)
> - **[wave-engine](https://github.com/atech-hub/wave-engine)** — Production training engine (Apache 2.0)
> - **[kerr-memory](https://github.com/atech-hub/kerr-memory)** — Wave memory state management (Apache 2.0, works with both servers)

---

OpenAI-compatible API server for [Kerr-ODE](https://github.com/atech-hub/kerr-engine) models. Self-contained — no engine dependency, no GPU code. Load a trained checkpoint, serve it via HTTP, optionally accumulate wave memory across conversations. Any chat UI that speaks the OpenAI protocol connects without modification.

### Why we moved on

The kerr-server proved the complete serving pipeline — OpenAI API, SSE streaming, bearer auth, wave memory accumulation, LM Studio compatibility. But the model architecture evolved:

- **Sequential blocks** in kerr-server don't match wave-engine's **parallel blocks** (GPT-J). A model trained with parallel blocks served through sequential blocks produces wrong outputs.
- **Single maestro** doesn't match wave-engine's **dual maestro**. The pre-ODE + post-ODE coordination is architecturally different.
- **Standard QKV attention** doesn't match wave-engine's **harmonic coherence attention**. Phase-based scoring replaces dot-product attention entirely.
- **No KV-cache** — at 768-dim, generating each token requires a full forward pass over the entire context. Wave-server adds KV-cache with cached phase angles, making generation practical at scale.

The server infrastructure (axum, API types, SSE streaming, bearer auth, memory integration, BPE tokenizer) was carried forward directly into wave-server. Only the model forward pass and checkpoint loader changed.

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

**LM Studio (verified):** Tested and verified with LM Studio 0.4.6. Set "Override Base URL" to `http://127.0.0.1:8080/v1`.

**Open WebUI:** Settings → Connections → add `http://127.0.0.1:8080/v1` as an OpenAI endpoint.

**SillyTavern / continue.dev / any OpenAI client:** Point the API base URL to `http://127.0.0.1:8080/v1`.

## Wave Memory

The `--memory` flag enables persistent experience accumulation across conversations. The model weights never change — a separate file (typically 1.5KB) shifts the Kerr-ODE's starting position on the unit circle.

Validated findings: stochastic resonance (α=0.05 improves perplexity by 8.8%), stable accumulation over 20 conversations, bit-identical reset on deletion, anomaly detection catches spikes before affecting output.

See [kerr-memory](https://github.com/atech-hub/kerr-memory) for the library, CLI tools, and full investigation results.

## Architecture

~1,900 lines across 10 modules. Self-contained — the forward pass is built in, no kerr-engine dependency.

**Model structure:** 4 blocks, each with CausalSelfAttention + FFN (Block 0: PerBandLinear, Blocks 1-3: KerrMaestroAdd). Sequential block formulation (GPT-2 style).

**Note:** This architecture does NOT match wave-engine's parallel block / dual-maestro / harmonic coherence architecture. For serving wave-engine models, use [wave-server](https://github.com/atech-hub/wave-server).

## Dependencies

- [kerr-memory](https://github.com/atech-hub/kerr-memory) — wave memory state management
- [axum](https://github.com/tokio-rs/axum) 0.8 — HTTP framework
- [tokio](https://tokio.rs) — async runtime
- serde, serde_json, uuid, tokio-stream

No kerr-engine dependency. No GPU code. No wgpu. No shaders.

## Contributing

This repository is no longer under active development. Bug fixes are welcome. For new feature development, please contribute to [wave-server](https://github.com/atech-hub/wave-server).

The maintainer (Marco Da Cunha) is an IT systems administrator, not a programmer. The server was built through collaboration with AI (Claude Desktop for architecture, Claude Code for implementation). This is stated openly.

## Related

- **[wave-server](https://github.com/atech-hub/wave-server)** — Production inference server (successor to this repo)
- **[wave-engine](https://github.com/atech-hub/wave-engine)** — Production training engine
- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project (public, MIT)
- [kerr-memory](https://github.com/atech-hub/kerr-memory) — Wave memory library (public, Apache 2.0)
- [Kerr Engine](https://github.com/atech-hub/kerr-engine) — Training engine for this server's checkpoints (historical)

## Credits

- **Marco Da Cunha** — Architecture, direction, pattern recognition
- **Claude Desktop (Opus)** — Architecture design, documentation
- **Claude Code** — Implementation, testing

## License

Apache 2.0. See [LICENSE](LICENSE).
