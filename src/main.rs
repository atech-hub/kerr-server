//! kerr-server — OpenAI-compatible API for Kerr-ODE models.
//!
//! Usage:
//!   kerr-server <checkpoint> <data> [options]
//!
//! v2 checkpoints self-describe — just point at the file.
//! v1 checkpoints need architecture flags to match the training config.

mod api_types;
mod handlers;
mod inference;
mod prompt;
mod server;

use std::sync::Arc;

use kerr_engine::checkpoint;
use kerr_engine::data::Dataset;
use kerr_engine::model::ModelConfig;

use crate::prompt::Vocab;
use crate::server::{AppState, ServerConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 || args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("kerr-server — OpenAI-compatible API for Kerr-ODE models\n");
        eprintln!("Usage: kerr-server <checkpoint> <data> [options]\n");
        eprintln!("Arguments:");
        eprintln!("  <checkpoint>      Path to checkpoint file (.bin)");
        eprintln!("  <data>            Path to training data (for vocabulary)\n");
        eprintln!("Options:");
        eprintln!("  --port N          Listen port (default: 8080)");
        eprintln!("  --host ADDR       Bind address (default: 127.0.0.1)");
        eprintln!("  --model-name S    Model name in responses (default: kerr-ode)");
        eprintln!("  --api-key KEY     Require Bearer token auth (default: none)");
        eprintln!("  --word            Use word-level tokenizer\n");
        eprintln!("Architecture (v1 checkpoints only — v2 self-describes):");
        eprintln!("  --n-bands N       Harmonic frequency bands (default: 64)");
        eprintln!("  --n-head N        Attention heads (default: 4)");
        eprintln!("  --n-layers N      Transformer blocks (default: 4)");
        eprintln!("  --maestro-dim N   Maestro bottleneck width (default: 16)");
        eprintln!("  --block-size N    Max sequence length (default: 256)");
        eprintln!("  --rk4-steps N     ODE integration steps (default: 8)");
        std::process::exit(1);
    }

    let checkpoint_path = &args[1];
    let data_path = &args[2];

    // Parse optional flags
    let mut port: u16 = 8080;
    let mut host = "127.0.0.1".to_string();
    let mut model_name = "kerr-ode".to_string();
    let mut api_key: Option<String> = None;
    let mut word_level = false;
    let mut config = ModelConfig::default_128();
    let mut has_arch_flags = false;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => { i += 1; port = args[i].parse().expect("invalid port"); }
            "--host" => { i += 1; host = args[i].clone(); }
            "--model-name" => { i += 1; model_name = args[i].clone(); }
            "--api-key" => { i += 1; api_key = Some(args[i].clone()); }
            "--word" => { word_level = true; }
            "--n-bands" => { i += 1; config.n_bands = args[i].parse().expect("invalid n-bands"); has_arch_flags = true; }
            "--n-head" => { i += 1; config.n_head = args[i].parse().expect("invalid n-head"); has_arch_flags = true; }
            "--n-layers" => { i += 1; config.n_layers = args[i].parse().expect("invalid n-layers"); has_arch_flags = true; }
            "--maestro-dim" => { i += 1; config.maestro_dim = args[i].parse().expect("invalid maestro-dim"); has_arch_flags = true; }
            "--block-size" => { i += 1; config.block_size = args[i].parse().expect("invalid block-size"); has_arch_flags = true; }
            "--rk4-steps" => { i += 1; config.rk4_n_steps = args[i].parse().expect("invalid rk4-steps"); has_arch_flags = true; }
            other => { eprintln!("Unknown flag: {other}"); std::process::exit(1); }
        }
        i += 1;
    }

    config.validate();

    // Load model from checkpoint
    // v2 checkpoints auto-detect config; v1 uses CLI flags or default_128
    println!("Loading checkpoint: {checkpoint_path}");
    let state = if has_arch_flags {
        checkpoint::load_with_config(checkpoint_path, config)
    } else {
        checkpoint::load(checkpoint_path)
    };
    let state = state.expect("Failed to load checkpoint");
    let model = state.model;
    println!("  Model: {} layers, {} embd ({} bands), {} vocab, {} params",
        model.config.n_layers, model.config.n_embd(), model.config.n_bands,
        model.vocab_size, kerr_engine::optim::count_params(&model));

    // Load dataset for vocabulary
    println!("Loading vocabulary from: {data_path}");
    let dataset = if word_level {
        Dataset::from_file_words(data_path, 0.9, 1)
    } else {
        Dataset::from_file(data_path)
    };
    let vocab = Vocab::from_dataset(&dataset);
    println!("  Vocab: {} tokens, mode={}",
        vocab.vocab_size, if word_level { "word" } else { "char" });

    // Verify vocab sizes match
    if vocab.vocab_size != model.vocab_size {
        eprintln!("WARNING: vocab size mismatch — model={}, data={}",
            model.vocab_size, vocab.vocab_size);
        eprintln!("  Model was trained with different data. Results may be incorrect.");
    }

    let app_state = Arc::new(AppState {
        model: Arc::new(model),
        vocab: Arc::new(vocab),
        config: ServerConfig { host, port, model_name, api_key },
    });

    // Start tokio runtime and run server
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(server::run(app_state));
}
