#!/usr/bin/env python3
"""
Simple CLI for recommendations.

Supports:
- recommend by `--product-id` (uses hybrid recommender if available, else TF-IDF)
- recommend by `--query` (uses TF-IDF on titles)

Example:
  python cli_recommender.py --product-id B08YRWN3WB --top-n 10
  python cli_recommender.py --query "blue jeans" --top-n 5
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import json
import requests
from io import BytesIO
from PIL import Image
import numpy as np


def _safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


def load_products(csv_path: str):
    if not Path(csv_path).exists():
        print(f"Products CSV not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    return df


def recommend_by_query(df, query, top_n=5):
    try:
        from processing.tfidf_title_similarity import tfidf_cosine_sim
    except Exception:
        print("TF-IDF module not available. Cannot run query-based recommendations.")
        return []

    titles = df['title'].fillna("").tolist()
    # Add query as a temporary last entry to compute similarity with titles
    corpus = titles + [query]
    # compute similarities where the last index is query
    sim_scores = tfidf_cosine_sim(idx=len(corpus)-1, n=top_n, products=corpus)
    recs = []
    for i, score in sim_scores:
        # skip the query self if present
        if i >= len(titles):
            continue
        recs.append((df.iloc[i]['product_id'], df.iloc[i]['title'], float(score)))
    return recs


def recommend_by_product(df, product_id, top_n=5):
    # Try to use HybridRecommender if available
    try:
        from hybrid_recommender_example import HybridRecommender
        recommender = HybridRecommender(csv_path=None)
        recommender.products_df = df
        recommender.tfidf_matrix = None
        recommender.prepare_data(process_images=False, download_images=False)
        # find index
        idx = df.index[df['product_id'] == product_id].tolist()
        if not idx:
            print(f"Product id {product_id} not found in CSV.")
            return []
        idx = idx[0]
        recs = recommender.get_hybrid_recommendations(product_idx=idx, top_n=top_n)
        # recs are tuples (index, score)
        out = []
        for r_idx, score in recs:
            out.append((df.iloc[r_idx]['product_id'], df.iloc[r_idx]['title'], float(score)))
        return out
    except Exception:
        # Fallback to TF-IDF
        titles = df['title'].fillna("").tolist()
        # find index
        idxs = df.index[df['product_id'] == product_id].tolist()
        if not idxs:
            print(f"Product id {product_id} not found in CSV.")
            return []
        idx = idxs[0]
        try:
            from processing.tfidf_title_similarity import tfidf_cosine_sim
            sim_scores = tfidf_cosine_sim(idx=idx, n=top_n, products=titles)
            out = []
            for i, score in sim_scores:
                out.append((df.iloc[i]['product_id'], df.iloc[i]['title'], float(score)))
            return out
        except Exception:
            print("No recommendation backend available (Hybrid + TF-IDF failed).")
            return []


def recommend_by_image_url(df, image_url, top_n=5):
    """Embed an image URL and find nearest products.

    Strategy:
    - Try to compute CLIP embedding locally (transformers+torch).
    - Then search using FAISS if available and index exists.
    - Otherwise load `data/embeddings/clip_embeddings.npy` and `data/embeddings/embedding_metadata.json`
      and compute cosine similarities.
    """
    # Download image
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to download/open image: {e}")
        return []

    # Try CLIP model
    transformers = _safe_import("transformers")
    torch = _safe_import("torch")
    image_embedding = None
    if transformers and torch:
        try:
            from transformers import CLIPProcessor, CLIPModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            inputs = processor(images=img, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            image_embedding = feats.cpu().numpy()[0]
            # normalize
            norm = np.linalg.norm(image_embedding)
            if norm > 0:
                image_embedding = image_embedding / norm
        except Exception as e:
            print(f"CLIP embedding failed: {e}")

    # If no CLIP model, try to preprocess and hope embeddings file contains preprocessed arrays
    if image_embedding is None:
        try:
            img_small = img.resize((224, 224), Image.BICUBIC)
            arr = np.array(img_small).astype("float32") / 255.0
            image_embedding = arr.reshape(-1)  # fallback: flattened array
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return []

    # Search for nearest embeddings
    # Try FAISS
    faiss = _safe_import("faiss")
    index_path = Path("data/indexes/hnsw_clip_512d.index")
    if faiss and index_path.exists():
        try:
            import faiss
            index = faiss.read_index(str(index_path))
            # embeddings must be float32 and L2 normalized
            q = np.array([image_embedding], dtype="float32")
            D, I = index.search(q, top_n)
            ids_path = Path("data/indexes/product_ids.json")
            if ids_path.exists():
                ids = json.load(open(ids_path))
            else:
                ids = None
            recs = []
            for i, d in zip(I[0], D[0]):
                if i < 0:
                    continue
                pid = ids[i] if ids else str(i)
                row = df[df['product_id'] == pid]
                title = row.iloc[0]['title'] if not row.empty else ""
                recs.append((pid, title, float(1.0 - d)))
            return recs
        except Exception as e:
            print(f"FAISS search failed: {e}")

    # Fallback: load embeddings numpy and metadata
    emb_path = Path("data/embeddings/clip_embeddings.npy")
    meta_path = Path("data/embeddings/embedding_metadata.json")
    if not emb_path.exists() or not meta_path.exists():
        print("No embeddings found. Generate embeddings first (see QUICK_START).")
        return []

    embs = np.load(str(emb_path))
    meta = json.load(open(str(meta_path)))
    ids = meta.get("ids", [])

    # If embs is images preprocessed arrays (4D), flatten appropriately
    if embs.ndim == 4:
        embs_proc = embs.reshape(embs.shape[0], -1)
        # L2 normalize
        norms = np.linalg.norm(embs_proc, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs_norm = embs_proc / norms
    else:
        embs_norm = embs

    q = image_embedding.astype("float32")
    # if q is flattened and embs are CLIP dims, try to reduce by PCA? For now compute cosine with what we have
    if embs_norm.ndim == 2 and q.ndim == 1 and embs_norm.shape[1] == q.shape[0]:
        sims = embs_norm @ q
    else:
        # try to project or compute using broadcast (fallback may be poor)
        sims = np.zeros(embs_norm.shape[0], dtype="float32")
        try:
            for i in range(embs_norm.shape[0]):
                a = embs_norm[i]
                denom = (np.linalg.norm(a) * np.linalg.norm(q))
                sims[i] = float(a.dot(q) / (denom if denom > 0 else 1.0))
        except Exception:
            print("Incompatible embedding shapes; cannot compute similarity.")
            return []

    # top-n
    idxs = np.argsort(-sims)[:top_n]
    recs = []
    for i in idxs:
        pid = ids[i] if i < len(ids) else str(i)
        row = df[df['product_id'] == pid]
        title = row.iloc[0]['title'] if not row.empty else ""
        recs.append((pid, title, float(sims[i])))
    return recs


def print_recs(recs):
    if not recs:
        print("No recommendations.")
        return
    print("Top recommendations:")
    for pid, title, score in recs:
        print(f"- {pid}: {title[:80]} (score: {score:.4f})")


def main():
    parser = argparse.ArgumentParser(description="CLI recommender: product-id or query")
    parser.add_argument("--csv", default="data/products.csv", help="Products CSV path")
    parser.add_argument("--product-id", help="Recommend for this product id")
    parser.add_argument("--query", help="Recommend for this text query")
    parser.add_argument("--image-url", help="Recommend for this image URL (requires CLIP embeddings or model)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations")

    args = parser.parse_args()

    df = load_products(args.csv)

    if args.product_id:
        recs = recommend_by_product(df, args.product_id, top_n=args.top_n)
        print_recs(recs)
    elif args.query:
        recs = recommend_by_query(df, args.query, top_n=args.top_n)
        print_recs(recs)
    elif args.image_url:
        recs = recommend_by_image_url(df, args.image_url, top_n=args.top_n)
        print_recs(recs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
