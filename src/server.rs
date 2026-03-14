//! HTTP server setup — Axum router, shared state, bind, graceful shutdown.

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tokio::net::TcpListener;

use kerr_engine::model::ModelWeights;

use crate::handlers;
use crate::prompt::Vocab;

/// Shared application state, injected into all handlers via Arc.
pub struct AppState {
    pub model: Arc<ModelWeights>,
    pub vocab: Arc<Vocab>,
    pub config: ServerConfig,
}

/// Server configuration from CLI.
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
}

/// Start the HTTP server with graceful shutdown on Ctrl+C.
pub async fn run(state: Arc<AppState>) {
    let host = state.config.host.clone();
    let port = state.config.port;

    let app = Router::new()
        .route("/v1/chat/completions", post(handlers::handle_chat_completion))
        .route("/v1/models", get(handlers::handle_models))
        .route("/health", get(handlers::handle_health))
        .with_state(state);

    let addr = format!("{host}:{port}");

    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ERROR: cannot bind to {addr} — {e}");
            if e.kind() == std::io::ErrorKind::AddrInUse {
                eprintln!("  Another process is using port {port}.");
                eprintln!("  Either stop it or use --port <other>");
            }
            std::process::exit(1);
        }
    };

    println!("  Server listening on http://{addr}");
    println!("  Endpoints:");
    println!("    POST /v1/chat/completions");
    println!("    GET  /v1/models");
    println!("    GET  /health");
    println!("  Press Ctrl+C to stop");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    println!("  Server stopped.");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    println!("\n  Shutting down...");
}
