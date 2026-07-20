mod confusion;
mod predict;

use anyhow::{Context, Result, anyhow};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use clap::Parser;
use confusion::{ConfusionMatrixResponse, load_confusion_matrix};
use mnist::MnistArtifacts;
use mnist::StandardizationParams;
use neural_network::Network;
use predict::{PredictRequest, predict_from_pixels, standardization_params_from_network};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use tower_http::services::ServeDir;

struct AppState {
    network: Network,
    standardization: StandardizationParams,
    confusion_matrix: Option<ConfusionMatrixResponse>,
}

#[derive(Parser, Debug)]
#[command(
    name = "mnist-demo",
    about = "Local MNIST drawing demo with live inference"
)]
struct Args {
    #[arg(long, default_value_t = 8765)]
    port: u16,

    #[arg(long, help = "Open the demo in the default browser")]
    open: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let workspace_root = std::env::current_dir().context("failed to get current directory")?;
    let artifacts = MnistArtifacts::from_workspace_root(&workspace_root);

    if !artifacts.model_path.exists() {
        anyhow::bail!(
            "trained model not found at {}\n\
             Train first from the workspace root:\n\
             cargo run --bin mnist --release -- train",
            artifacts.model_path.display()
        );
    }

    let model_path = artifacts
        .model_path
        .to_str()
        .ok_or_else(|| anyhow!("invalid model path"))?;

    println!("Loading model from {}...", artifacts.model_path.display());
    let network = Network::load(model_path).context("failed to load trained network")?;
    let standardization = standardization_params_from_network(&network)?;
    let confusion_matrix = load_confusion_matrix(&network);

    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");
    let state = Arc::new(AppState {
        network,
        standardization,
        confusion_matrix,
    });

    let app = Router::new()
        .route("/predict", post(predict_handler))
        .route("/confusion-matrix", get(confusion_matrix_handler))
        .fallback_service(ServeDir::new(static_dir))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let url = format!("http://{addr}");
    println!("MNIST demo listening on {url}");

    if args.open {
        open_browser(&url)?;
    }

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    axum::serve(listener, app)
        .await
        .context("server exited with error")?;

    Ok(())
}

async fn confusion_matrix_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match &state.confusion_matrix {
        Some(matrix) => (StatusCode::OK, Json(matrix)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "MNIST test set not available. Add mnist/data/ to show the confusion matrix."
            })),
        )
            .into_response(),
    }
}

async fn predict_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictRequest>,
) -> impl IntoResponse {
    match predict_from_pixels(&state.network, &state.standardization, &request.pixels) {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": error.to_string() })),
        )
            .into_response(),
    }
}

fn open_browser(url: &str) -> Result<()> {
    Command::new("open")
        .arg(url)
        .status()
        .context("failed to launch browser with `open`")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_dir_exists() {
        let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");
        assert!(static_dir.join("index.html").is_file());
        assert!(static_dir.join("index.js").is_file());
    }
}
