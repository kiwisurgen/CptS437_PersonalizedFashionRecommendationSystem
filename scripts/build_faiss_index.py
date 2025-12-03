"""Build FAISS HNSW index from saved embeddings and metadata.
Saves index to `data/indexes/hnsw_clip_<dim>.faiss` and metadata `.pkl` via FAISSIndex.save,
and also writes a JSON product_id mapping for CLI convenience.
"""
import os
import json
import numpy as np
from pathlib import Path
import sys
# Ensure repo root is on sys.path so local packages (evaluation) can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.ann_indexing import FAISSIndex

EMB_PATH = Path("data/embeddings/clip_embeddings.npy")
META_PATH = Path("data/embeddings/embedding_metadata.json")
OUT_DIR = Path("data/indexes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not EMB_PATH.exists():
    raise FileNotFoundError(f"Embeddings file not found: {EMB_PATH}")
if not META_PATH.exists():
    raise FileNotFoundError(f"Embedding metadata file not found: {META_PATH}")

print(f"Loading embeddings from {EMB_PATH}...")
embeddings = np.load(str(EMB_PATH))
print(f"Embeddings shape: {embeddings.shape}")

print(f"Loading metadata from {META_PATH}...")
with open(META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)

# metadata may use key 'ids' or 'product_ids'
product_ids = meta.get('ids') or meta.get('product_ids')
if product_ids is None:
    raise KeyError("Could not find 'ids' or 'product_ids' in metadata JSON")

if len(product_ids) != embeddings.shape[0]:
    print("Warning: number of product_ids != number of embeddings. Truncating to min length.")
    n = min(len(product_ids), embeddings.shape[0])
    product_ids = product_ids[:n]
    embeddings = embeddings[:n]

dim = embeddings.shape[1]
base_name = f"hnsw_clip_{dim}"
out_path = OUT_DIR / base_name

print(f"Building FAISS HNSW index (dim={dim})...")
index = FAISSIndex(dimension=dim, index_type='HNSW', metric='IP')
index.build(embeddings, product_ids, normalize=True)

print(f"Saving index to {out_path} (.faiss + .pkl)")
index.save(str(out_path))

# Also write product ids mapping JSON for CLI
product_ids_json = OUT_DIR / f"{base_name}_product_ids.json"
with open(product_ids_json, 'w', encoding='utf-8') as f:
    json.dump({'product_ids': product_ids}, f)

print(f"Wrote product_ids JSON to {product_ids_json}")
print("Done.")
