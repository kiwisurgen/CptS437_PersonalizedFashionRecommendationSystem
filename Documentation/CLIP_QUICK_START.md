CLIP Embedding Quick Start
==========================

This short guide shows how to generate CLIP image embeddings (sample and full runs) and where outputs are written. It assumes the repository root is the current working directory.

Prerequisites
- A Python virtual environment (recommended).
- `pip` packages: `pandas`, `numpy`, `Pillow`, `requests`, `transformers`, and a `torch` build compatible with your CUDA driver (if you want GPU acceleration).

Install example (CUDA 13.0 example — adjust for your system):

```powershell
# create and activate virtualenv (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# core deps
pip install -r requirements.txt

# Example: install a CUDA 13.0-compatible PyTorch wheel (replace with the official recommended command for your platform)
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision --upgrade

# install transformers (if not already)
pip install transformers
```

Quick checks

```powershell
# Check torch + GPU availability
python -c "import torch,sys; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())"
```

Sample run (quick validation)

This runs the generator in "sample" mode (uses a small set or cached images) and writes a small embeddings file.

```powershell
python .\generate_image_embeddings.py --sample 100 --download-missing --batch-size 8 --output data/embeddings/clip_embeddings_sample.npy --meta data/embeddings/embedding_metadata_sample.json
```

Full run (entire catalog)

This runs over all products in `data/products.csv`. It will skip images already cached in `data/image_cache` and resume cleanly if interrupted.

```powershell
python .\generate_image_embeddings.py --sample 0 --download-missing --batch-size 16 --output data/embeddings/clip_embeddings.npy --meta data/embeddings/embedding_metadata.json
```

Outputs
- Embeddings: `data/embeddings/clip_embeddings.npy` (float32, shape: N x D)
- Metadata: `data/embeddings/embedding_metadata.json` (list of product ids / ordering)

Tips & troubleshooting
- If you see the script fallback to saving preprocessed images rather than embeddings, your environment likely lacks `transformers` or a working `torch` install. Re-run in a venv where those are installed.
- The script caches downloaded images under `data/image_cache/<product_id>.<ext>` — re-running skips already-cached images.
- To reduce network load, run with `--download-missing` only once; subsequent runs will reuse cache.

Next steps
- After embeddings are created, consider building a FAISS index (see `evaluation/ann_indexing.py`) or run the CLI (`cli_recommender.py`) which can use the embeddings for image-based search.

