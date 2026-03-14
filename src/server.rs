//! HTTP server setup — Axum router, shared state, auth, graceful shutdown.

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Json, Response};
use tokio::net::TcpListener;

use kerr_engine::model::ModelWeights;

use crate::api_types::{ErrorResponse, ErrorDetail};
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
    pub api_key: Option<String>,
}

/// Bearer token auth middleware.
async fn auth_middleware(
    State(state): axum::extract::State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let Some(ref expected_key) = state.config.api_key else {
        // No key configured — allow all requests
        return next.run(req).await;
    };

    let auth_header = req.headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            if token == expected_key {
                return next.run(req).await;
            }
        }
        _ => {}
    }

    let err = ErrorResponse {
        error: ErrorDetail {
            message: "Invalid or missing API key. Use Authorization: Bearer <key>".to_string(),
            r#type: "authentication_error".to_string(),
        },
    };
    (StatusCode::UNAUTHORIZED, Json(err)).into_response()
}

use axum::extract::State;

/// Start the HTTP server with graceful shutdown on Ctrl+C.
pub async fn run(state: Arc<AppState>) {
    let host = state.config.host.clone();
    let port = state.config.port;
    let has_auth = state.config.api_key.is_some();

    // Protected routes (require API key if configured)
    let protected = Router::new()
        .route("/v1/chat/completions", post(handlers::handle_chat_completion))
        .route("/v1/models", get(handlers::handle_models))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // Public routes (health check always open for load balancers)
    let app = Router::new()
        .merge(protected)
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
    println!("  Auth: {}", if has_auth { "API key required" } else { "NONE (use --api-key to secure)" });
    println!("  Endpoints:");
    println!("    POST /v1/chat/completions{}", if has_auth { " [auth]" } else { "" });
    println!("    GET  /v1/models{}", if has_auth { " [auth]" } else { "" });
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
