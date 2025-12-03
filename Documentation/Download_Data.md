Download Data — instructions
============================

Purpose
- This file explains which large data artifacts are intentionally ignored by Git and how to provide them (e.g., via Google Drive). Place the files into these exact paths in your repo after downloading.

Ignored paths (matches `.gitignore`)
- `data/image_cache/` — directory containing downloaded product images (many files, e.g. `12345.jpg`).
- `data/embeddings/*.npy` — NumPy embedding arrays (e.g. `clip_embeddings.npy`, `clip_embeddings_sample.npy`).
- `data/embeddings/*.json` — embedding metadata files that map embedding rows to `product_id` (e.g. `embedding_metadata.json`).

Recommended file names & locations
- `data/embeddings/clip_embeddings.npy`
- `data/embeddings/embedding_metadata.json`
- `data/embeddings/clip_embeddings_sample.npy` (optional sample)
- `data/embeddings/embedding_metadata_sample.json` (optional sample)
- `data/image_cache/` — a flat directory containing files named by product id (the pipeline accepts typical image extensions).


Sharing / Links
- https://drive.google.com/drive/folders/1LEN5xZ_MBZDiP2_iUPyhCZpTYoe3OyCZ?usp=drive_link

Downloading from Google Drive (recommendations)
Option A — Browser manual download
- Open the shared folder in Google Drive and download the files (or the zipped `image_cache`).
- Place the files into your local repo under the exact paths listed above.

Important notes
- Do NOT commit these files to Git. They are listed in `.gitignore` to avoid large blobs in the repository.
- If you must version them, consider `git lfs` or an artifact store (S3, Azure Blob, etc.).
- The embedding files are typically float32 NumPy arrays. Keep `embedding_metadata.json` in sync with the ordering of rows.
- `data/image_cache/` should contain image files named using product ids (or the naming the pipeline expects). The pipeline will skip re-downloading files that already exist in the cache.

Troubleshooting
- If your scripts report missing embeddings, confirm `data/embeddings/clip_embeddings.npy` and `data/embeddings/embedding_metadata.json` exist and are readable.
- If image-based code fails, check that `data/image_cache/` contains images and has read permissions.
- For partial re-runs, the embedding generator is resume-safe: it re-uses cached images and will compute embeddings for missing items.



