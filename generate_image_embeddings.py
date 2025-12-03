#!/usr/bin/env python3
"""
Generate image embeddings for product catalog using CLIP (transformers) if available.

Usage examples:
  # generate embeddings for first 100 products (download images if missing)
  python generate_image_embeddings.py --sample 100 --download-missing --output data/embeddings/clip_embeddings_sample.npy

If `transformers` and `torch` are not installed, the script will save preprocessed image arrays
to `--output` (numpy .npy) and print instructions to install CLIP dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import math


def safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate image embeddings for products using CLIP (optional)")
    parser.add_argument("--csv", default="data/products.csv", help="Products CSV path")
    parser.add_argument("--sample", type=int, default=0, help="Process only first N products (0 = all)")
    parser.add_argument("--output", default="data/embeddings/clip_embeddings.npy", help="Output .npy path")
    parser.add_argument("--meta", default="data/embeddings/embedding_metadata.json", help="Metadata JSON path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="HuggingFace CLIP model name")
    parser.add_argument("--download-missing", action="store_true", help="Download images that are missing from cache")
    parser.add_argument("--cache-dir", default="data/image_cache", help="Local image cache directory")
    args = parser.parse_args(argv)

    # Lazy imports
    pd = safe_import("pandas")
    if pd is None:
        print("pandas is required to run this script. Install with `pip install pandas`.")
        sys.exit(1)

    # Load products
    df = pd.read_csv(args.csv)
    if args.sample and args.sample > 0:
        df = df.head(args.sample)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.meta), exist_ok=True)

    # Determine which products have cached images
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    products = []
    for _, row in df.iterrows():
        pid = str(row.get("product_id") or row.get("id") or row.name)
        cached = cache_dir / f"{pid}.jpg"
        products.append({
            "product_id": pid,
            "image_url": row.get("image_url"),
            "title": row.get("title", ""),
            "cached_path": str(cached) if cached.exists() else None,
        })

    # Optionally download missing images using local processing utilities if available
    if args.download_missing:
        # Try to import our project's ImageEmbeddingProcessor
        try:
            from processing.image_embedding import ImageEmbeddingProcessor

            processor = ImageEmbeddingProcessor(cache_dir=str(cache_dir))
        except Exception:
            processor = None

        for p in products:
            if p["cached_path"] is None:
                url = p["image_url"]
                pid = p["product_id"]
                if not url:
                    continue
                if processor is not None:
                    try:
                        res = processor.process_image_url(url, pid)
                        if res and res.get("saved_path"):
                            p["cached_path"] = res.get("saved_path")
                    except Exception as e:
                        print(f"Failed to download/process {pid}: {e}")
                else:
                    # Simple download fallback
                    try:
                        from PIL import Image
                        import requests
                        from io import BytesIO

                        resp = requests.get(url, timeout=8)
                        resp.raise_for_status()
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                        target = cache_dir / f"{pid}.jpg"
                        img.save(target, format="JPEG", quality=90)
                        p["cached_path"] = str(target)
                    except Exception as e:
                        print(f"Fallback download failed for {pid}: {e}")

    # Filter to only products with cached images
    products_with_images = [p for p in products if p["cached_path"]]
    if len(products_with_images) == 0:
        print("No cached images found. Use --download-missing to fetch images or populate data/image_cache.")
        sys.exit(1)

    # Try to load CLIP via transformers
    transformers = safe_import("transformers")
    torch = safe_import("torch")

    if transformers and torch:
        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            model = CLIPModel.from_pretrained(args.model).to(device)
            processor = CLIPProcessor.from_pretrained(args.model)

            embeddings = []
            ids = []
            batch = []
            batch_ids = []
            total = len(products_with_images)
            for idx, p in enumerate(products_with_images, start=1):
                img = Image.open(p["cached_path"]).convert("RGB")
                batch.append(img)
                batch_ids.append(p["product_id"])

                if len(batch) >= args.batch_size or idx == total:
                    inputs = processor(images=batch, return_tensors="pt")
                    # move tensors to device
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)
                    with torch.no_grad():
                        out = model.get_image_features(**inputs)
                    arr = out.cpu().numpy()
                    embeddings.append(arr)
                    ids.extend(batch_ids)
                    batch = []
                    batch_ids = []

            embeddings = np.vstack(embeddings)
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

            np.save(args.output, embeddings)
            meta = {
                "model": args.model,
                "count": len(ids),
                "ids": ids,
            }
            with open(args.meta, "w", encoding="utf8") as fh:
                json.dump(meta, fh, indent=2)

            print(f"Saved embeddings to {args.output} (shape: {embeddings.shape})")
            print(f"Saved metadata to {args.meta}")
            return
        except Exception as e:
            print(f"Error using transformers CLIP backend: {e}")

    # Fallback: save preprocessed numpy arrays for later embedding
    print("transformers/torch not available or failed — saving preprocessed image arrays instead.")
    from PIL import Image

    arrs = []
    ids = []
    for p in products_with_images:
        try:
            img = Image.open(p["cached_path"]).convert("RGB")
            img = img.resize((224, 224), Image.BICUBIC)
            a = np.array(img).astype("float32") / 255.0
            arrs.append(a)
            ids.append(p["product_id"])
        except Exception as e:
            print(f"Failed to process {p['product_id']}: {e}")

    arrs = np.stack(arrs, axis=0)
    np.save(args.output, arrs)
    meta = {"model": None, "count": len(ids), "ids": ids, "note": "Preprocessed arrays saved; install torch+transformers to compute CLIP embeddings."}
    with open(args.meta, "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved preprocessed arrays to {args.output} (shape: {arrs.shape})")
    print(f"Saved metadata to {args.meta}")


if __name__ == "__main__":
    main()
