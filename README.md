# CptS437 Personalized Fashion Recommendation System

A comprehensive multimodal fashion recommendation system that combines text-based and image-based similarity for intelligent product recommendations.

## 🎯 Project Overview

This system processes fashion product data and uses both **text embeddings** (TF-IDF) and **image embeddings** (multimodal models like CLIP) to provide personalized recommendations.

### Key Features
- ✅ **Multimodal Processing**: Combine text and image data for recommendations
- ✅ **Image URL Processing**: Validate, download, and cache product images
- ✅ **TF-IDF Text Similarity**: Fast text-based product matching
- ✅ **Embedding Ready**: Compatible with CLIP, ViT, and other vision models
- ✅ **Batch Processing**: Efficiently handle large product catalogs
- ✅ **Hybrid Recommendations**: Weighted combination of text and image similarity

## 📁 Project Structure

```
CptS437_PersonalizedFashionRecommendationSystem/
├── preprocessing/
│   └── preprocess_product_data.py      # Data cleaning and image URL validation
├── processing/
│   ├── tfidf_title_similarity.py       # Text-based similarity using TF-IDF
│   └── image_embedding.py              # Image processing for embeddings
├── data/
│   ├── products.csv                    # Product catalog (13,000+ items)
│   └── image_cache/                    # Downloaded product images
├── test_image_pipeline.py              # Verification tests
├── hybrid_recommender_example.py       # Integration example
├── IMAGE_PROCESSING.md                 # Image processing guide
├── INTEGRATION_SUMMARY.md              # Implementation overview
└── requirements.txt                    # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

For image embeddings with transformers:
```bash
pip install torch torchvision transformers
```

### 2. Validate Image URLs
```python
from preprocessing.preprocess_product_data import preprocess_fashion_data

df = preprocess_fashion_data(
    csv_path="data/products.csv",
    process_images=True,
    download_images=False
)
print(f"Valid URLs: {df['image_url_valid'].sum()}/{len(df)}")
```

### 3. Download Images
```python
df = preprocess_fashion_data(
    csv_path="data/products.csv",
    process_images=True,
    download_images=True,
    image_cache_dir="data/image_cache"
)
```

### 4. Get Recommendations
```python
from hybrid_recommender_example import HybridRecommender

recommender = HybridRecommender(csv_path="data/products.csv")
recommender.prepare_data()

# Text-based recommendations
text_recs = recommender.compute_text_similarities(product_idx=5, top_n=5)

# Hybrid recommendations (when embeddings available)
hybrid_recs = recommender.get_hybrid_recommendations(
    product_idx=5,
    top_n=5,
    text_weight=0.5,
    image_weight=0.5
)
```

## 📊 Data Pipeline

```
┌─────────────────────────┐
│  CSV: products.csv      │ (13,000+ products)
│ - title, brand, price   │
│ - rating, image_url     │
└────────────┬────────────┘
             │
    ┌────────▼────────┐
    │ Preprocessing   │
    │ - Validate URLs │
    │ - Remove NaN    │
    │ - Deduplication │
    └────────┬────────┘
             │
    ┌────────▼────────────────┐
    │ Text Embeddings         │
    │ TF-IDF on titles        │
    │ Similarity: cosine      │
    └────────────┬────────────┘
             │
    ┌────────▼─────────────────┐
    │ Image Embeddings         │
    │ Download & process URLs  │
    │ Vision model (CLIP/ViT)  │
    │ Similarity: cosine       │
    └────────────┬─────────────┘
             │
    ┌────────▼────────────────┐
    │ Hybrid Recommendations  │
    │ Combine scores (weighted)│
    │ Rank & return           │
    └────────────────────────┘
```

## 🔧 Core Modules

### `preprocess_product_data.py`
Handles data cleaning and image URL processing:
- `preprocess_fashion_data()` - Main preprocessing function
- `validate_image_url()` - Check if URL is valid
- `download_image()` - Fetch image from URL
- `process_image_urls()` - Batch process URLs

**Usage:**
```python
df = preprocess_fashion_data(
    csv_path="data/products.csv",
    process_images=True,
    download_images=True,
    image_cache_dir="data/image_cache"
)
```

### `tfidf_title_similarity.py`
Text-based similarity using TF-IDF:
- `tfidf_cosine_sim()` - Compute similarity scores
- `top_n_similar()` - Get top N similar products

**Usage:**
```python
from processing.tfidf_title_similarity import tfidf_cosine_sim

products = df['title'].tolist()
similarities = tfidf_cosine_sim(idx=0, n=5, products=products)
```

### `image_embedding.py`
Image processing and embedding preparation:
- `ImageEmbeddingProcessor` - Main processor class
  - `download_image_from_url()` - Download images
  - `preprocess_image()` - Resize and normalize
  - `batch_process_images()` - Process multiple URLs
  - `load_cached_image()` - Load preprocessed image
  - `get_batch_embeddings()` - Generate embeddings

**Usage:**
```python
from processing.image_embedding import ImageEmbeddingProcessor

processor = ImageEmbeddingProcessor(cache_dir="data/image_cache")
processor.batch_process_images(df)
image_array = processor.load_cached_image("product_id")
```

## 📚 Documentation

- **[IMAGE_PROCESSING.md](IMAGE_PROCESSING.md)** - Comprehensive image processing guide
- **[INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)** - Implementation overview
- **[hybrid_recommender_example.py](hybrid_recommender_example.py)** - Full integration example

## 🧪 Testing

Run the verification tests:
```bash
python test_image_pipeline.py
```

Tests include:
1. URL validation
2. CSV image processing
3. Single image download
4. Batch processing
5. Cache loading

## 💡 Integration Examples

### Text-Only Recommendations
```python
from processing.tfidf_title_similarity import tfidf_cosine_sim

products = df['title'].tolist()
sims = tfidf_cosine_sim(idx=5, n=10, products=products)
for product_idx, score in sims:
    print(f"{df.iloc[product_idx]['title']}: {score:.4f}")
```

### With CLIP Embeddings
```python
import torch
from transformers import CLIPModel, CLIPProcessor
from processing.image_embedding import ImageEmbeddingProcessor

processor = ImageEmbeddingProcessor()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Get embeddings and recommendations
embeddings = processor.get_batch_embeddings(product_ids, model)
```

### Hybrid Approach
```python
from hybrid_recommender_example import HybridRecommender

recommender = HybridRecommender("data/products.csv")
recs = recommender.get_hybrid_recommendations(
    product_idx=5,
    text_weight=0.4,      # 40% text similarity
    image_weight=0.6      # 60% image similarity
)
```

## 📦 Dependencies

**Required:**
- pandas ≥ 1.3.0
- scikit-learn ≥ 0.24.0
- numpy ≥ 1.21.0
- Pillow ≥ 9.0.0
- requests ≥ 2.28.0

**Optional (for embeddings):**
- torch ≥ 1.9.0
- torchvision ≥ 0.10.0
- transformers ≥ 4.20.0

See [requirements.txt](requirements.txt) for complete list.

## 🎯 Workflow Example

```python
# 1. Load and preprocess
from preprocessing.preprocess_product_data import preprocess_fashion_data
df = preprocess_fashion_data("data/products.csv", process_images=True, download_images=True)

# 2. Find similar products (text-based)
from processing.tfidf_title_similarity import tfidf_cosine_sim
text_sims = tfidf_cosine_sim(idx=0, n=5, products=df['title'].tolist())

# 3. Find similar products (image-based) - with embeddings
from processing.image_embedding import ImageEmbeddingProcessor
processor = ImageEmbeddingProcessor()
# Load your embedding model and generate recommendations

# 4. Combine approaches for hybrid recommendations
from hybrid_recommender_example import HybridRecommender
recommender = HybridRecommender("data/products.csv")
hybrid_recs = recommender.get_hybrid_recommendations(0, top_n=10)
```

## 🔌 Embedding Model Integration

The system is designed to work with any embedding model:

```python
def generate_embeddings(image_array):
    """Your embedding model wrapper"""
    # Load image into your model
    # Return embedding vector
    pass

# Use with processor
embeddings = processor.get_batch_embeddings(product_ids, generate_embeddings)
```

Compatible models:
- CLIP (OpenAI)
- Vision Transformer (ViT)
- ResNet
- EfficientNet
- And more!

## 🚦 Performance

- **URL validation**: ~100-200 URLs/sec
- **Image download**: ~5-10 images/sec (network dependent)
- **TF-IDF similarity**: <1ms per query
- **Batch embedding**: Model dependent (GPU recommended)

## 📝 Notes

- Images are cached to `data/image_cache/` after download
- Cached images are normalized to [0,1] and resized to 224×224
- Use `skip_existing=True` in batch processing to avoid reprocessing
- GPU recommended for batch embedding generation

## 📄 Data Format

### Input (CSV)
```csv
product_id,brand,title,price,category,rating,image_url,product_url
B08YRWN3WB,JANSPORT,Big Student Backpack,189.0,New season,4.7,https://...,https://...
```

### Preprocessed Output (with images)
```
product_id | brand | title | price | ... | image_url_valid | image_local_path
B08YRWN3WB | ... | ... | ... | ... | True | data/image_cache/B08YRWN3WB.jpg
```

## 🤝 Contributing

Areas for improvement:
- [ ] Multi-language text support
- [ ] Attribute-based filtering
- [ ] User preference learning
- [ ] Real-time recommendations
- [ ] API endpoint creation

## 📜 License

CptS437 Course Project

## ✅ Status

✅ Image processing pipeline implemented
✅ Text similarity functional
✅ Multimodal integration ready
🔜 Embedding models integration (next phase)
🔜 API deployment (production phase)

---

**Last Updated:** November 30, 2025
**Branch:** main
**Version:** 1.0 - Image Processing Release