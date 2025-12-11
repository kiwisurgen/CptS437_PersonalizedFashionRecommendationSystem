"""Utilities to load precomputed embeddings, metadata, and FAISS indexes.

This module centralizes logic for finding/loading:
- NumPy embeddings (`data/embeddings/clip_embeddings.npy`)
- Embedding metadata JSON (`data/embeddings/embedding_metadata.json`)
- FAISS index files under `data/indexes/` (pick the first `.faiss` found)

Functions return None when resources are not available so callers can fallback.
"""
from pathlib import Path
import json
import numpy as np
from typing import Tuple, Optional, List


def load_embeddings(emb_path: Path = Path("data/embeddings/clip_embeddings.npy"),
                    meta_path: Path = Path("data/embeddings/embedding_metadata.json")) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Load embeddings and metadata ids.

    Returns (embeddings, ids) or (None, None) if files are missing or invalid.
    """
    emb_path = Path(emb_path)
    meta_path = Path(meta_path)
    if not emb_path.exists() or not meta_path.exists():
        return None, None
    try:
        embs = np.load(str(emb_path))
    except Exception:
        return None, None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        ids = meta.get('ids') or meta.get('product_ids')
    except Exception:
        ids = None
    return embs, ids


def load_faiss_index(index_dir: Path = Path("data/indexes")) -> Tuple[Optional[object], Optional[List[str]]]:
    """Attempt to load a FAISS index and its product-id mapping.

    Returns (faiss_index, product_ids) or (None, None) if not available.
    The function selects the first `.faiss` file found in `index_dir`.
    """
    try:
        import faiss
    except Exception:
        return None, None

    index_dir = Path(index_dir)
    if not index_dir.exists():
        return None, None

    faiss_files = list(index_dir.glob("*.faiss"))
    if not faiss_files:
        return None, None

    faiss_path = faiss_files[0]
    try:
        index = faiss.read_index(str(faiss_path))
    except Exception:
        return None, None

    # try a few strategies to find product ids mapping
    prefix = faiss_path.stem  # e.g. hnsw_clip_512
    # JSON with product ids
    pid_json = index_dir / f"{prefix}_product_ids.json"
    if pid_json.exists():
        try:
            with open(pid_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            product_ids = data.get('product_ids') or data.get('ids')
            return index, product_ids
        except Exception:
            pass

    # try pkl metadata (same prefix).pkl created by FAISSIndex.save
    pkl_path = index_dir / f"{prefix}.pkl"
    if pkl_path.exists():
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                meta = pickle.load(f)
            product_ids = meta.get('product_ids')
            return index, product_ids
        except Exception:
            pass

    # fallback: None for product ids
    return index, None
