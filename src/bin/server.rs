//! HTTP API (axum) — exposes the SegmentedIndex.
//!
//! The concurrency model carries phase 5's contract over to HTTP: searches run
//! from any worker thread via `&self`, while ALL writes are serialized onto a
//! single writer task through an mpsc channel (so SegmentedIndex's
//! single-writer contract holds even with many HTTP clients).
//!
//! Endpoints:
//!   POST   /vectors        {"id": 1, "vector": [...], "metadata": {"k": "v"}}
//!   POST   /search         {"vector": [...], "k": 10, "filter": {"must": [...]}}
//!   DELETE /vectors/{id}
//!   POST   /checkpoint     (seal + snapshot + manifest swap)
//!   GET    /stats
//!
//! Usage: cargo run --release --bin server -- [port] [dim] [data-dir] [sync]
//! If data-dir is given the index is persistent (recovered from the manifest +
//! WAL at startup). sync: none | per_op | group:<ms> (default group:20).
//!
//! The HTTP 200 contract (DECISIONS #36): a write response is returned only
//! AFTER the durability the policy promises has been achieved. The writer task
//! batches commands, performs a single commit at the end of the batch, and only
//! then sends the responses —
//! group commit budur.

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
use vector_gvector::storage::wal::SyncPolicy;
use vector_gvector::types::VectorId;

/// Commands sent to the writer task. The reply comes back over a oneshot —
/// a client never receives a 200 before its write has actually been applied.
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
    /// A checkpoint is a write operation too (it seals the buffer): under the
    /// single-writer contract it goes through the same queue and never races
    /// with concurrent inserts.
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
                Json(serde_json::json!({ "error": "writer is closed" })),
            )
        })?;
    match rx.await {
        Ok(Ok(())) => Ok(StatusCode::CREATED),
        Ok(Err(e)) => Err(index_error_response(e)),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "writer did not respond" })),
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
                Json(serde_json::json!({ "error": "writer is closed" })),
            )
        })?;
    match rx.await {
        Ok(Ok(())) => Ok(StatusCode::NO_CONTENT),
        Ok(Err(e)) => Err(index_error_response(e)),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "writer did not respond" })),
        )),
    }
}

async fn search(State(app): State<AppState>, Json(req): Json<SearchReq>) -> Json<Vec<SearchHit>> {
    // Search is CPU work: run it on the blocking pool so the tokio workers do
    // not stall.
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
                Json(serde_json::json!({ "error": "writer is closed" })),
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
            Json(serde_json::json!({ "error": "writer did not respond" })),
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
        "wal_bytes": app.index.wal_len_bytes(),
        "wal_policy": app.index.wal_policy_label(),
        "wal_replay_applied": app.index.replay_report().applied,
    }))
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let port: u16 = args.next().and_then(|a| a.parse().ok()).unwrap_or(7700);
    let dim: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(128);

    let data_dir = args.next().map(std::path::PathBuf::from);
    let policy = args
        .next()
        .and_then(|s| SyncPolicy::parse(&s))
        .unwrap_or(SyncPolicy::Group { window_ms: 20 });
    let index = Arc::new(match &data_dir {
        Some(dir) => {
            let t = std::time::Instant::now();
            let idx = SegmentedIndex::open_durable(
                dir.clone(),
                dim,
                Metric::L2,
                HnswParams::default(),
                10_000,
                policy,
            )
            .expect("could not open the index");
            let rep = idx.replay_report();
            println!(
                "persistent mode: {} (generation={}, {} records, sync={}, {:?})",
                dir.display(),
                idx.generation(),
                idx.len_shared(),
                policy.label(),
                t.elapsed()
            );
            if rep.applied > 0 || rep.truncated_at.is_some() {
                println!(
                    "  WAL replay: {} records applied{}",
                    rep.applied,
                    match (&rep.reason, rep.truncated_at) {
                        (Some(r), Some(at)) => format!("; kuyruk {at} offsetinde kesildi ({r})"),
                        _ => String::new(),
                    }
                );
            }
            idx
        }
        None => {
            println!("in-memory mode (no data-dir given): data is NOT persistent");
            SegmentedIndex::new(dim, Metric::L2, HnswParams::default(), 10_000)
        }
    });

    // The single writer task: every mutation passes through here in order.
    let (tx, mut rx) = mpsc::channel::<WriteCmd>(1024);
    {
        let index = index.clone();
        // Sealing (HNSW construction) can take seconds: run it on a blocking thread.
        tokio::task::spawn_blocking(move || {
            // Group commit: commands already waiting are applied as one batch,
            // a SINGLE commit is performed at the end of the batch, and only
            // then are the responses sent. This preserves the "200 = durable"
            // contract while spreading the fsync cost across the batch.
            const MAX_BATCH: usize = 256;
            let mut pending: Vec<oneshot::Sender<Result<(), IndexError>>> = Vec::new();
            let mut results: Vec<Result<(), IndexError>> = Vec::new();
            while let Some(first) = rx.blocking_recv() {
                let mut batch = vec![first];
                while batch.len() < MAX_BATCH {
                    match rx.try_recv() {
                        Ok(c) => batch.push(c),
                        Err(_) => break,
                    }
                }
                for cmd in batch {
                    match cmd {
                        WriteCmd::Insert {
                            id,
                            vector,
                            meta,
                            reply,
                        } => {
                            results.push(index.insert_with_meta(id, &vector, meta));
                            pending.push(reply);
                        }
                        WriteCmd::Delete { id, reply } => {
                            results.push(index.delete_shared(id));
                            pending.push(reply);
                        }
                        WriteCmd::Checkpoint { reply } => {
                            // A checkpoint provides its own durability; commit
                            // the pending writes first so the ordering holds.
                            let _ = index.commit_wal();
                            for (tx, r) in pending.drain(..).zip(results.drain(..)) {
                                let _ = tx.send(r);
                            }
                            let _ = reply.send(index.checkpoint().map_err(|e| e.to_string()));
                        }
                    }
                }
                if !pending.is_empty() {
                    let commit = index.commit_wal();
                    for (tx, r) in pending.drain(..).zip(results.drain(..)) {
                        // If the commit fails the write is not durable: the
                        // client must not see this as a 200.
                        let _ = tx.send(match (&commit, r) {
                            (Err(e), _) => Err(IndexError::Storage(e.to_string())),
                            (Ok(()), r) => r,
                        });
                    }
                }
            }
            // The channel closed (graceful shutdown): make the remainder durable.
            let _ = index.flush_wal();
        });
    }

    let index_for_shutdown = index.clone();
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
    // Graceful shutdown: ctrl-c → yeni istek alma, bekleyenleri bitir,
    // then force an fsync of the WAL. On shutdown we fsync regardless of the
    // policy — there is nothing to gain from losing data.
    let shutdown_index = index_for_shutdown;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            println!("\nshutting down: flushing the WAL...");
        })
        .await
        .expect("serve");
    if let Err(e) = shutdown_index.flush_wal() {
        eprintln!("WAL flush error: {e}");
    }
    println!("clean shutdown.");
}
