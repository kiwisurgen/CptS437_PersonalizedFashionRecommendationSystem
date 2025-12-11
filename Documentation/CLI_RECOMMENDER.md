# CLI Recommender — Usage Guide

This document explains how to use the command-line interface `cli_recommender.py` included in this repository.

**File:** `cli_recommender.py`

---

## Prerequisites

- Python environment with the project `requirements.txt` installed (recommended to use the repo `.venv`).
- Optional for image queries: `torch`, `transformers` (CLIP). If not installed, the CLI will try to use precomputed embeddings or fall back to a numpy search if embeddings exist.
- Data: `data/products.csv` must exist and be readable by the script.

Run the repo Python from PowerShell (example):

```powershell
# Activate your venv if you have one (example)
& .\.venv\Scripts\Activate.ps1
# Or explicitly use the venv python to run commands
& .\.venv\Scripts\python.exe .\cli_recommender.py --help
```

---

## Basic flags

- `--csv`: Path to products CSV (default: `data/products.csv`).
- `--product-id`: Recommend for a specific product id (uses Hybrid recommender if available, otherwise TF‑IDF fallback).
- `--query`: Text query; runs TF‑IDF similarity on titles.
- `--image-url`: Recommend from an image URL (requires CLIP or precomputed embeddings + FAISS/numpy fallback).
- `--top-n`: Number of recommendations to return (default: `5`).
- `--format`: `text` (default) or `json` output.
- `--interactive`: Start an interactive prompt mode.

Example help call:

```powershell
python .\cli_recommender.py --help
```

---

## Quick Examples

Text-mode (TF-IDF):

```powershell
python .\cli_recommender.py --query "blue denim jacket" --top-n 5
```

Product-id (hybrid fallback -> TF-IDF):

```powershell
python .\cli_recommender.py --product-id B08YRWN3WB --top-n 8 --format text
```

Image URL (CLIP/FAISS preferred; falls back to numpy embeddings if FAISS unavailable):

```powershell
python .\cli_recommender.py --image-url "https://example.com/image.jpg" --top-n 6 --format json
```

Interactive mode (menu-driven):

```powershell
python .\cli_recommender.py --interactive
```

- Interactive flow prompts for mode (product id / text query / image URL), `top-n` and output format.
- Type `q` or `quit` at any prompt to exit.

---

## Output Formats

- `text`: Nicely formatted table with Rank, Product ID, Score, and truncated Title.
- `json`: Machine-readable JSON array with `rank`, `product_id`, `title`, `score` and optional fields from the CSV (if present): `brand`, `price`, `category`, `product_url`.

---

## Image Query Fallbacks & Notes

- If `transformers` + `torch` are available, the CLI will attempt to compute a CLIP embedding locally and search the FAISS index (if present).
- If FAISS is not available or index not present, the script falls back to loading precomputed numpy embeddings (`processing.embedding_loader.load_embeddings`) and performs a brute-force cosine search.
- If no embeddings or index are available, the CLI will prompt you to generate embeddings and build the FAISS index; see `scripts/build_faiss_index.py` and `generate_image_embeddings.py`.

---

## Troubleshooting

- "Products CSV not found": Ensure `--csv` points to the correct file and you run the command from the repo root.
- Argparse errors like `unrecognized arguments: -- top-n 5`: use `--top-n 5` (no space between `--` and the flag name).
- For image queries: network download issues will be reported; verify the URL is reachable.

---

## Automation / Scripts

You can run the CLI from scripts or pipelines. Example PowerShell non-interactive run:

```powershell
& .\.venv\Scripts\python.exe .\cli_recommender.py --image-url "https://...jpg" --top-n 5 --format json > results.json
```

This writes JSON-formatted results to `results.json` for downstream processing.

---

## Where to look in the code

- Interactive & CLI parsing: `cli_recommender.py`
- TF-IDF similarity: `processing/tfidf_title_similarity.py`
- Image preprocessing & embeddings: `processing/image_embedding.py`, `generate_image_embeddings.py`
- FAISS index build: `scripts/build_faiss_index.py`, `evaluation/ann_indexing.py`
- Hybrid ranking example: `hybrid_recommender_example.py`

---

If you want, I can add a short link from the root `README.md` to this file or show an example recorded interactive session output to include in docs.
