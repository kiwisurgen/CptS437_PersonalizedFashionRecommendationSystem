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
import textwrap
import requests
from io import BytesIO
from PIL import Image
import numpy as np


def _safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


# local loader for embeddings/index
try:
    from processing.embedding_loader import load_embeddings, load_faiss_index
except Exception:
    load_embeddings = None
    load_faiss_index = None


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
    # 1) Try FAISS index via loader (preferred)
    if load_faiss_index is not None:
        try:
            index_obj, pid_list = load_faiss_index()
        except Exception:
            index_obj, pid_list = None, None
    else:
        index_obj, pid_list = None, None

    if index_obj is not None:
        try:
            import faiss
            q = np.array([image_embedding], dtype="float32")
            # ensure shape
            if q.ndim == 1:
                q = q.reshape(1, -1)
            D, I = index_obj.search(q, top_n)
            recs = []
            for idx, dist in zip(I[0], D[0]):
                if idx < 0:
                    continue
                pid = pid_list[idx] if (pid_list and idx < len(pid_list)) else str(idx)
                row = df[df['product_id'] == pid]
                title = row.iloc[0]['title'] if not row.empty else ""
                # For inner-product on normalized vectors, higher is better (dist is similarity)
                # faiss returns distances consistent with index metric; for HNSW+IP dist is similarity
                score = float(dist)
                recs.append((pid, title, score))
            return recs
        except Exception as e:
            print(f"FAISS search failed: {e}")

    # 2) Fallback: load numpy embeddings and metadata via loader
    embs, ids = (None, None)
    if load_embeddings is not None:
        try:
            embs, ids = load_embeddings()
        except Exception:
            embs, ids = None, None

    if embs is None or ids is None:
        print("No embeddings/index found. Generate embeddings first (see QUICK_START).")
        return []

    # Flatten/prep embeddings if necessary and normalize for cosine
    if embs.ndim == 4:
        embs_proc = embs.reshape(embs.shape[0], -1)
    else:
        embs_proc = embs

    # L2-normalize embeddings and query
    embs_norm = embs_proc.astype('float32')
    norms = np.linalg.norm(embs_norm, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = embs_norm / norms

    q = image_embedding.astype('float32')
    if q.ndim == 1:
        q = q.reshape(-1)
    q_norm = q / (np.linalg.norm(q) if np.linalg.norm(q) > 0 else 1.0)

    if embs_norm.shape[1] != q_norm.shape[0]:
        print("Embedding dimension mismatch between query and stored embeddings; cannot compute similarity.")
        return []

    sims = embs_norm @ q_norm
    idxs = np.argsort(-sims)[:top_n]
    recs = []
    for i in idxs:
        pid = ids[i] if i < len(ids) else str(i)
        row = df[df['product_id'] == pid]
        title = row.iloc[0]['title'] if not row.empty else ""
        recs.append((pid, title, float(sims[i])))
    return recs


def _format_recs_text(recs, df=None, title_width=60):
    """Return a pretty text table for recommendations.

    Columns: Rank | Product ID | Score | Title
    """
    if not recs:
        return "No recommendations."

    lines = []
    hdr_rank = "#"
    hdr_pid = "Product ID"
    hdr_score = "Score"
    hdr_title = "Title"

    # column widths
    w_rank = 4
    w_pid = 14
    w_score = 8
    w_title = title_width

    header = f"{hdr_rank:>{w_rank}}  {hdr_pid:<{w_pid}}  {hdr_score:>{w_score}}  {hdr_title:<{w_title}}"
    sep = f"{'-' * w_rank}  {'-' * w_pid}  {'-' * w_score}  {'-' * w_title}"
    lines.append(header)
    lines.append(sep)

    for i, (pid, title, score) in enumerate(recs, start=1):
        short_title = textwrap.shorten(str(title), width=title_width, placeholder="...")
        lines.append(f"{i:>{w_rank}}.  {str(pid):<{w_pid}}  {score:>{w_score}.4f}  {short_title:<{w_title}}")

    return "\n".join(lines)


def _format_recs_json(recs, df=None):
    out = []
    for i, (pid, title, score) in enumerate(recs, start=1):
        entry = {"rank": i, "product_id": pid, "title": title, "score": score}
        # optionally enrich with df info if provided
        if df is not None:
            row = df[df['product_id'] == pid]
            if not row.empty:
                r = row.iloc[0]
                for col in ['brand', 'price', 'category', 'product_url']:
                    if col in r:
                        entry[col] = r[col]
        out.append(entry)
    return json.dumps(out, indent=2, ensure_ascii=False)


def print_recs(recs, df=None, out_format='text'):
    if not recs:
        print("No recommendations.")
        return

    if out_format == 'json':
        print(_format_recs_json(recs, df=df))
    else:
        print(_format_recs_text(recs, df=df))


def main():
    parser = argparse.ArgumentParser(description="CLI recommender: product-id or query")
    parser.add_argument("--csv", default="data/products.csv", help="Products CSV path")
    parser.add_argument("--product-id", help="Recommend for this product id")
    parser.add_argument("--query", help="Recommend for this text query")
    parser.add_argument("--image-url", help="Recommend for this image URL (requires CLIP embeddings or model)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--format", choices=['text', 'json'], default='text', help="Output format")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt mode")

    args = parser.parse_args()

    df = load_products(args.csv)

    def _interactive_loop():
        """Run a simple interactive prompt loop for the CLI."""
        print("Interactive recommender — enter 'q' at any prompt to quit")
        while True:
            print("\nSelect mode:\n  1) product id\n  2) text query\n  3) image URL\n  4) quit")
            choice = input("Choice [1-4]: ").strip()
            if choice.lower() in ['4', 'q', 'quit', 'exit']:
                print("Exiting interactive mode.")
                break

            if choice not in ['1', '2', '3']:
                print("Invalid choice — please enter 1, 2, 3 or 4.")
                continue

            top_n_str = input(f"Top N results (default {args.top_n}): ").strip()
            if top_n_str.lower() in ['q', 'quit', 'exit']:
                break
            try:
                top_n = int(top_n_str) if top_n_str else args.top_n
            except Exception:
                print("Invalid number, using default.")
                top_n = args.top_n

            out_fmt = input(f"Output format ('text' or 'json', default {args.format}): ").strip()
            if out_fmt.lower() in ['q', 'quit', 'exit']:
                break
            if out_fmt not in ['text', 'json', '']:
                print("Unknown format, using default.")
                out_fmt = args.format
            out_fmt = out_fmt if out_fmt else args.format

            if choice == '1':
                pid = input("Enter product id: ").strip()
                if pid.lower() in ['q', 'quit', 'exit']:
                    break
                recs = recommend_by_product(df, pid, top_n=top_n)
                print_recs(recs, df=df, out_format=out_fmt)
            elif choice == '2':
                q = input("Enter text query: ").strip()
                if q.lower() in ['q', 'quit', 'exit']:
                    break
                recs = recommend_by_query(df, q, top_n=top_n)
                print_recs(recs, df=df, out_format=out_fmt)
            elif choice == '3':
                url = input("Enter image URL: ").strip()
                if url.lower() in ['q', 'quit', 'exit']:
                    break
                recs = recommend_by_image_url(df, url, top_n=top_n)
                print_recs(recs, df=df, out_format=out_fmt)

    # If interactive flag set, or no mode args provided and running in a TTY, enter interactive mode
    if args.interactive or (not (args.product_id or args.query or args.image_url) and sys.stdin.isatty()):
        try:
            _interactive_loop()
        except (KeyboardInterrupt, EOFError):
            print("\nInteractive session ended.")
        return

    if args.product_id:
        recs = recommend_by_product(df, args.product_id, top_n=args.top_n)
        print_recs(recs, df=df, out_format=args.format)
    elif args.query:
        recs = recommend_by_query(df, args.query, top_n=args.top_n)
        print_recs(recs, df=df, out_format=args.format)
    elif args.image_url:
        recs = recommend_by_image_url(df, args.image_url, top_n=args.top_n)
        print_recs(recs, df=df, out_format=args.format)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
