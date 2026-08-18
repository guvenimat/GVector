//! HTTP API (axum) — SegmentedIndex'i dışa açar.
//!
//! Eşzamanlılık modeli Aşama 5'in sözleşmesini HTTP'ye taşır: aramalar
//! herhangi bir worker thread'inden `&self` ile koşar; TÜM yazmalar tek bir
//! yazıcı task'ine mpsc kanalıyla sıralanır (SegmentedIndex'in tek-yazar
//! sözleşmesi HTTP istemcileri çok olsa da korunur).
//!
//! Uç noktalar:
//!   POST   /vectors        {"id": 1, "vector": [...], "metadata": {"k": "v"}}
//!   POST   /search         {"vector": [...], "k": 10, "filter": {"must": [...]}}
//!   DELETE /vectors/{id}
//!   POST   /checkpoint     (mühürle + snapshot + manifest takası)
//!   GET    /stats
//!
//! Kullanım: cargo run --release --bin server -- [port] [dim] [data-dir]
//! data-dir verilirse indeks kalıcıdır (açılışta manifest'ten kurtarılır).

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use vector_gvector::distance::Metric;
use vector_gvector::index::hnsw::HnswParams;
use vector_gvector::index::segmented::SegmentedIndex;
use vector_gvector::index::IndexError;
use vector_gvector::meta::{Filter, Metadata};
use vector_gvector::types::VectorId;

/// Yazıcı task'ine gönderilen komutlar. Yanıt oneshot ile geri döner —
/// istemci, yazması gerçekten uygulanmadan 200 almaz.
enum WriteCmd {
    Insert {
        id: VectorId,
        vector: Vec<f32>,
        meta: Metadata,
        reply: oneshot::Sender<Result<(), IndexError>>,
    },
    Delete {
        id: VectorId,
        reply: oneshot::Sender<Result<(), IndexError>>,
    },
    /// Checkpoint de bir yazma işidir (buffer'ı mühürler): tek yazar
    /// sözleşmesi gereği aynı kuyruktan geçer, eşzamanlı insert'lerle
    /// yarışmaz.
    Checkpoint {
        reply: oneshot::Sender<Result<u64, String>>,
    },
}

#[derive(Clone)]
struct AppState {
    index: Arc<SegmentedIndex>,
    writer: mpsc::Sender<WriteCmd>,
}

#[derive(serde::Deserialize)]
struct InsertReq {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: Metadata,
}

#[derive(serde::Deserialize)]
struct SearchReq {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Filter,
}

fn default_k() -> usize {
    10
}

#[derive(serde::Serialize)]
struct SearchHit {
    id: u64,
    distance: f32,
}

fn index_error_response(e: IndexError) -> (StatusCode, Json<serde_json::Value>) {
    let code = match e {
        IndexError::NotFound(_) => StatusCode::NOT_FOUND,
        IndexError::DuplicateId(_) => StatusCode::CONFLICT,
        _ => StatusCode::BAD_REQUEST,
    };
    (code, Json(serde_json::json!({ "error": e.to_string() })))
}

async fn insert_vector(
    State(app): State<AppState>,
    Json(req): Json<InsertReq>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let (tx, rx) = oneshot::channel();
    app.writer
        .send(WriteCmd::Insert {
            id: VectorId(req.id),
            vector: req.vector,
            meta: req.metadata,
            reply: tx,
        })
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "yazıcı kapalı" })),
            )
        })?;
    match rx.await {
        Ok(Ok(())) => Ok(StatusCode::CREATED),
        Ok(Err(e)) => Err(index_error_response(e)),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "yazıcı yanıt vermedi" })),
        )),
    }
}

async fn delete_vector(
    State(app): State<AppState>,
    Path(id): Path<u64>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let (tx, rx) = oneshot::channel();
    app.writer
        .send(WriteCmd::Delete {
            id: VectorId(id),
            reply: tx,
        })
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "yazıcı kapalı" })),
            )
        })?;
    match rx.await {
        Ok(Ok(())) => Ok(StatusCode::NO_CONTENT),
        Ok(Err(e)) => Err(index_error_response(e)),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "yazıcı yanıt vermedi" })),
        )),
    }
}

async fn search(State(app): State<AppState>, Json(req): Json<SearchReq>) -> Json<Vec<SearchHit>> {
    // Arama CPU işi: blocking havuzunda koş ki tokio worker'ları tıkanmasın.
    let index = app.index.clone();
    let hits = tokio::task::spawn_blocking(move || {
        index
            .search_filtered(&req.vector, req.k, &req.filter)
            .into_iter()
            .map(|r| SearchHit {
                id: r.id.0,
                distance: r.distance,
            })
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();
    Json(hits)
}

async fn checkpoint(
    State(app): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let (tx, rx) = oneshot::channel();
    app.writer
        .send(WriteCmd::Checkpoint { reply: tx })
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "yazıcı kapalı" })),
            )
        })?;
    match rx.await {
        Ok(Ok(generation)) => Ok(Json(serde_json::json!({ "generation": generation }))),
        Ok(Err(e)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e })),
        )),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "yazıcı yanıt vermedi" })),
        )),
    }
}

async fn stats(State(app): State<AppState>) -> Json<serde_json::Value> {
    let (segments, buffer) = app.index.shape();
    Json(serde_json::json!({
        "len": app.index.len_shared(),
        "segments": segments,
        "buffer": buffer,
        "generation": app.index.generation(),
        "last_checkpoint_unix": app.index.last_checkpoint_unix(),
        "durable": app.index.storage_dir().is_some(),
        "storage_dir": app.index.storage_dir().map(|d| d.display().to_string()),
    }))
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let port: u16 = args.next().and_then(|a| a.parse().ok()).unwrap_or(7700);
    let dim: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(128);

    let data_dir = args.next().map(std::path::PathBuf::from);
    let index = Arc::new(match &data_dir {
        Some(dir) => {
            let t = std::time::Instant::now();
            let idx = SegmentedIndex::open_or_create(
                dir.clone(),
                dim,
                Metric::L2,
                HnswParams::default(),
                10_000,
            )
            .expect("indeks açılamadı");
            println!(
                "kalıcı mod: {} (generation={}, {} kayıt, {:?})",
                dir.display(),
                idx.generation(),
                idx.len_shared(),
                t.elapsed()
            );
            idx
        }
        None => {
            println!("bellek-içi mod (data-dir verilmedi): veriler kalıcı DEĞİL");
            SegmentedIndex::new(dim, Metric::L2, HnswParams::default(), 10_000)
        }
    });

    // Tek yazıcı task'i: tüm mutasyonlar buradan sırayla geçer.
    let (tx, mut rx) = mpsc::channel::<WriteCmd>(1024);
    {
        let index = index.clone();
        // Mühürleme (HNSW inşası) saniyeler sürebilir: blocking thread'de koş.
        tokio::task::spawn_blocking(move || {
            while let Some(cmd) = rx.blocking_recv() {
                match cmd {
                    WriteCmd::Insert {
                        id,
                        vector,
                        meta,
                        reply,
                    } => {
                        let _ = reply.send(index.insert_with_meta(id, &vector, meta));
                    }
                    WriteCmd::Delete { id, reply } => {
                        let _ = reply.send(index.delete_shared(id));
                    }
                    WriteCmd::Checkpoint { reply } => {
                        let _ = reply.send(index.checkpoint().map_err(|e| e.to_string()));
                    }
                }
            }
        });
    }

    let app = Router::new()
        .route("/vectors", post(insert_vector))
        .route("/vectors/:id", delete(delete_vector))
        .route("/search", post(search))
        .route("/checkpoint", post(checkpoint))
        .route("/stats", get(stats))
        .with_state(AppState { index, writer: tx });

    let addr = format!("127.0.0.1:{port}");
    println!("vector-gvector API dinliyor: http://{addr} (dim={dim})");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}
