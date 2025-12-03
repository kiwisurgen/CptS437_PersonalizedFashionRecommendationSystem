Image Processing — Addendum
===========================

This addendum documents the concrete image preprocessing, caching, and resume behaviour used by the repository's image pipeline.

Cache location
- All downloaded images are stored in `data/image_cache/`.
- File naming: `data/image_cache/<product_id>.<ext>` (the pipeline chooses a stable extension when saving).

Preprocessing details
- Images are opened with Pillow and converted to `RGB`.
- Images are resized and padded to a square default of `224x224` (configurable in code) to match CLIP preprocessing expectations.
- Pixel data is converted to `float32` and normalized to the [0,1] range before being converted to tensors by the model pipeline.

Resumability & idempotence
- The embedding generator skips image downloads if a cached file exists for a `product_id`.
- If the embedding script is interrupted, re-running with the same `--output` path will resume by reusing cached images; new embeddings will be computed for missing items.

How to re-run a failed job
1. Ensure your venv has `transformers` and a compatible `torch` build installed.
2. Re-run the exact `generate_image_embeddings.py` command you used previously — it will pick up from the cache and only process what remains.

CLI usage notes
- The command-line recommender supports image-based queries via `--image-url`. Example:

```powershell
python .\cli_recommender.py --image-url "https://example.com/image.jpg" --top-k 10
```

- If the CLI cannot find a FAISS index, it will fall back to a numpy-based nearest-neighbour search using `data/embeddings/clip_embeddings.npy`.

Performance & production tips
- For production/large-scale runs, use a GPU-backed `torch` install and increase `--batch-size` to improve throughput.
- For long-term storage of large embedding files, consider `git lfs` or a separate artifact store instead of committing to Git.

