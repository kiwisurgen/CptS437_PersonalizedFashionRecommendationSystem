# Image Processing for Multimodal Embeddings

This guide explains how to use the image processing functionality in the preprocessing and processing modules.

## Overview

The system provides two main components for image processing:

1. **`preprocess_product_data.py`** - Data cleaning and image URL validation
2. **`processing/image_embedding.py`** - Image downloading, caching, and embedding preparation

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

For deep learning embeddings, also install:

```bash
pip install torch torchvision transformers
```

## Usage

### 1. Basic Image URL Validation

Validate image URLs without downloading:

```python
from preprocessing.preprocess_product_data import preprocess_fashion_data

# Load and validate image URLs
df = preprocess_fashion_data(
    csv_path="data/products.csv",
    process_images=True,
    download_images=False
)

# Check validation results
print(f"Valid URLs: {df['image_url_valid'].sum()}")
print(f"Invalid URLs: {(~df['image_url_valid']).sum()}")
```

### 2. Download and Cache Images

Download all valid images locally:

```python
df = preprocess_fashion_data(
    csv_path="data/products.csv",
    process_images=True,
    download_images=True,
    image_cache_dir="data/image_cache"
)

# Check how many were cached
cached_count = df['image_local_path'].notna().sum()
print(f"Cached images: {cached_count}/{len(df)}")
```

### 3. Prepare Images for Embeddings

Use the `ImageEmbeddingProcessor` to prepare images:

```python
from processing.image_embedding import ImageEmbeddingProcessor
import pandas as pd

# Initialize processor
processor = ImageEmbeddingProcessor(
    cache_dir="data/image_cache",
    target_size=(224, 224)  # For most vision models
)

# Process single image
url = "https://example.com/image.jpg"
result = processor.process_image_url(url, product_id="prod_123")

# Result contains:
# - image_array: normalized numpy array ready for model input
# - cache_path: local path where image was saved
# - shape: image array shape (224, 224, 3)
```

### 4. Batch Process from DataFrame

```python
df = pd.read_csv("data/products.csv")

processed_df = processor.batch_process_images(
    df,
    url_column='image_url',
    id_column='product_id',
    skip_existing=True  # Skip if already cached
)

# Check results
successful = processed_df['image_processed'].sum()
print(f"Successfully processed: {successful}/{len(processed_df)}")
```

### 5. Load Cached Images for Embedding

```python
# Load single cached image
image_array = processor.load_cached_image(product_id="prod_123")
# image_array is now a numpy array of shape (224, 224, 3) with values in [0, 1]

# Batch load for embedding model
product_ids = df['product_id'].head(10).tolist()
embeddings = processor.get_batch_embeddings(
    product_ids,
    embedding_model=your_embedding_model
)
# embeddings shape: (10, embedding_dim)
```

## Image Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                 Image URL from CSV                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Validate URL & fetch  │
            └────────────┬───────────┘
                         │
            ┌────────────▼───────────────┐
            │   Verify valid image       │
            │  (content-type check)      │
            └────────────┬───────────────┘
                         │
            ┌────────────▼───────────────┐
            │  Download full image       │
            │  (with retry)              │
            └────────────┬───────────────┘
                         │
            ┌────────────▼───────────────┐
            │  Convert to RGB            │
            │  Resize to 224x224         │
            │  Normalize [0,1]           │
            └────────────┬───────────────┘
                         │
            ┌────────────▼───────────────┐
            │  Cache locally (JPG)       │
            │  product_id.jpg            │
            └────────────┬───────────────┘
                         │
            ┌────────────▼───────────────┐
            │  Return numpy array ready  │
            │  for embedding model       │
            └────────────────────────────┘
```

## Example: Complete Workflow

```python
from preprocessing.preprocess_product_data import preprocess_fashion_data
from processing.image_embedding import ImageEmbeddingProcessor
import pandas as pd

# Step 1: Preprocess and validate
print("Step 1: Preprocessing and validation...")
df = preprocess_fashion_data(
    csv_path="data/products.csv",
    process_images=True,
    download_images=False
)
valid_urls = df['image_url_valid'].sum()
print(f"  Valid URLs found: {valid_urls}/{len(df)}")

# Step 2: Download valid images
print("\nStep 2: Downloading and caching images...")
processor = ImageEmbeddingProcessor()
df_valid = df[df['image_url_valid']].copy()
processed_df = processor.batch_process_images(df_valid)
cached = processed_df['image_processed'].sum()
print(f"  Successfully cached: {cached}/{len(df_valid)}")

# Step 3: Ready for embedding models
print("\nStep 3: Images ready for multimodal embeddings!")
print(f"  Cache directory: {processor.cache_dir}")
print(f"  Image size: {processor.target_size}")
print(f"  Use load_cached_image() to load individual images")
```

## Image Array Format

Images are normalized to **[0, 1]** range as float32:

```python
# Image array properties:
image_array.dtype  # float32
image_array.shape  # (224, 224, 3) for RGB
image_array.min()  # 0.0
image_array.max()  # 1.0

# Convert back to uint8 if needed:
import numpy as np
uint8_image = (image_array * 255).astype(np.uint8)
```

## Integration with Embedding Models

### CLIP (OpenAI)

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from processing.image_embedding import ImageEmbeddingProcessor

processor_img = ImageEmbeddingProcessor()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embeddings(product_ids):
    embeddings = []
    for pid in product_ids:
        image = processor_img.load_cached_image(pid)
        if image is not None:
            # Add batch dimension
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model.get_image_features(image_tensor)
            embeddings.append(outputs.cpu().numpy())
    return embeddings
```

### Custom Vision Model

```python
def get_embeddings(product_ids, model):
    embeddings = processor.get_batch_embeddings(product_ids, model)
    return embeddings
```

## Configuration

### Image Size

Most pretrained models expect 224x224, but you can customize:

```python
processor = ImageEmbeddingProcessor(target_size=(384, 384))  # For ViT-base-384
```

### Cache Directory

```python
processor = ImageEmbeddingProcessor(cache_dir="models/image_cache")
```

### Timeout Settings

```python
from preprocessing.preprocess_product_data import validate_image_url
validate_image_url(url, timeout=10)  # Increase timeout for slow connections
```

## Error Handling

The system gracefully handles:
- ✅ Missing/invalid URLs
- ✅ Timeout errors during download
- ✅ Corrupted image files
- ✅ Unsupported image formats (auto-converts to RGB)
- ✅ Network errors with logging

Check logs for detailed error information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Use `skip_existing=True`** to avoid reprocessing cached images
2. **Batch process** for efficiency
3. **Consider image resolution** - 224x224 is standard, larger sizes use more memory
4. **Cache images locally** to avoid repeated downloads
5. **Use GPU** for embedding models when available

## Troubleshooting

### "Failed to download image from URL"
- Check if URL is publicly accessible
- Verify internet connection
- Increase timeout value
- Check if the URL is an Amazon link (may have rate limiting)

### "Failed to generate embedding"
- Ensure image is properly cached
- Verify embedding model is properly loaded
- Check GPU memory if using GPU

### Memory Issues with Large Batches
- Process smaller batches
- Use smaller image size (e.g., 224x224 instead of 384x384)
- Load only as many images as needed

## Next Steps

1. Run the preprocessing script to validate URLs
2. Download images for valid products
3. Integrate with your chosen embedding model (CLIP, ViT, ResNet, etc.)
4. Use embeddings for similarity calculations in recommendations
