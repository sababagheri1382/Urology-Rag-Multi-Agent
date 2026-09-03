import gc
import json
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "final_rag_collection"
DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5" 
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
BATCH_SIZE = 64
CHUNKS_FILE = "final_granular_chunks_all.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print(" Warning: GPU not available. This will be slow for Dense embeddings.")

client = QdrantClient(
    url="CLUSTER_URL",    
    api_key="API_KEY",    
    timeout=60.0              
)

print("Loading chunks...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    all_chunks = data.get("chunks", list(data.values()))
else:
    all_chunks = data

print(f"Loaded {len(all_chunks)} chunks.")

print("Loading Dense model (SentenceTransformer)...")
dense_model = SentenceTransformer(DENSE_MODEL_NAME, device=DEVICE)

print("Loading Sparse model...")
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in collections:
    print("Creating collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense-vector": VectorParams(size=1024, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse-vector": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True)
            )
        }
    )
    print("Collection created.")
else:
    print("Collection already exists. Will append new points.")


def get_existing_ids():
    ids = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_vectors=False,
            with_payload=False
        )
        for p in points:
            ids.add(p.id)
        if next_offset is None:
            break
        offset = next_offset
    return ids


existing_ids = get_existing_ids()
print(f"Existing points: {len(existing_ids)}")


def upsert_batch(batch):
    texts = [ch["text"] for ch in batch]

    # Dense Embedding روی GPU با نرمال‌سازی (COSINE)
    dense_embs = dense_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    sparse_embs = list(sparse_model.embed(texts))

    points = []
    for i, ch in enumerate(batch):
        sp_indices = sparse_embs[i].indices.tolist()
        sp_values = sparse_embs[i].values.tolist()

        points.append(PointStruct(
            id=ch["chunk_index"],
            vector={
                "dense-vector": dense_embs[i].tolist(),
                "sparse-vector": models.SparseVector(
                    indices=sp_indices,
                    values=sp_values
                )
            },
            payload={
                "chunk_id": ch.get("chunk_id", ""),
                "parent_chunk_id": ch.get("parent_chunk_id", ""),
                "doc_id": ch.get("doc_id", ""),
                "breadcrumb": ch.get("breadcrumb", ""),
                "text": ch["text"],
                "has_table": bool(ch.get("has_table", False)),
                "chunk_length": len(ch["text"])
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)


total = len(all_chunks)
skipped = 0

for i in range(0, total, BATCH_SIZE):
    batch = all_chunks[i:i + BATCH_SIZE]

    new_batch = [c for c in batch if c.get("chunk_index") not in existing_ids]

    if not new_batch:
        skipped += len(batch)
        print(f"Batch {i//BATCH_SIZE+1}: all {len(batch)} already exist. Skipping.")
        continue

    try:
        upsert_batch(new_batch)
        for c in new_batch:
            existing_ids.add(c["chunk_index"])
        print(f"Batch {i//BATCH_SIZE+1}/{(total-1)//BATCH_SIZE+1}: "
              f"upserted {len(new_batch)} new")
        gc.collect()
    except Exception as e:
        print(f"ERROR in batch {i}: {e}")
        with open("ingest_checkpoint.json", "w") as cf:
            json.dump({"last_index": i}, cf)
        break

print("=" * 50)
print(f"DONE. Upserted new chunks. Skipped {skipped} already-existing.")
