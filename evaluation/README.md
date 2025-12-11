# Evaluation Module

Complete evaluation framework for the fashion recommendation system.

## Components

### 📊 Metrics (`metrics.py`)

Standard information retrieval metrics:

- **Precision@K** - Proportion of relevant items in top K
- **Recall@K** - Proportion of relevant items found
- **NDCG@K** - Normalized Discounted Cumulative Gain
- **MRR** - Mean Reciprocal Rank
- **Hit Rate@K** - Proportion of queries with at least one hit
- **MAP@K** - Mean Average Precision

```python
from evaluation.metrics import RecommendationEvaluator

evaluator = RecommendationEvaluator(k_values=[5, 10, 20])
results = evaluator.evaluate(
    recommended_lists,
    relevant_items,
    relevance_scores=graded_relevance  # Optional
)
```

### 🎲 Baseline Recommenders (`baselines.py`)

Simple baseline systems for comparison:

1. **RandomRecommender** - Random recommendations (sanity check)
2. **PopularityRecommender** - Most popular items
3. **TFIDFRecommender** - Text similarity using TF-IDF
4. **CategoryBasedRecommender** - Same category + popularity

```python
from evaluation.baselines import TFIDFRecommender

recommender = TFIDFRecommender(df)
recs = recommender.recommend(product_id, top_n=10)
```

### ⚡ ANN Indexing (`ann_indexing.py`)

FAISS-based fast similarity search:

```python
from evaluation.ann_indexing import FAISSIndex

# Create index
index = FAISSIndex(dimension=512, index_type='HNSW')
index.build(embeddings, product_ids, normalize=True)

# Search
results = index.search(query_embedding, top_k=10)

# Benchmark
from evaluation.ann_indexing import benchmark_index
results = benchmark_index(embeddings, product_ids)
```

**Index Types:**
- **Flat** - Exact search (baseline, ~1ms for 13k items)
- **IVF** - Inverted file index (faster, ~0.5ms)
- **HNSW** - Hierarchical NSW (fastest, ~0.1ms)

## 📓 Evaluation Notebook

Run `evaluation_benchmark.ipynb` for complete evaluation:

1. Creates weak relevance labels from ratings
2. Evaluates all baseline recommenders
3. Benchmarks FAISS ANN indices
4. Generates visualizations
5. Exports report to `Documentation/EVALUATION.md`

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install FAISS for ANN search
pip install faiss-cpu

# Run Jupyter
jupyter notebook evaluation_benchmark.ipynb
```

## Labeling Scheme

Since explicit user data isn't available, we use **weak supervision**:

| Score | Criteria |
|-------|----------|
| **2** | Same category + same brand + high rating (≥4.5) |
| **1** | Same category OR (same brand + high rating) |
| **0** | Different category and lower rating |

This simulates implicit signals (clicks, views) in production.

## Expected Performance

**Baselines (13k products):**
- Random: NDCG@10 ≈ 0.05, Latency ~1ms
- Popularity: NDCG@10 ≈ 0.15, Latency ~1ms  
- TF-IDF: NDCG@10 ≈ 0.35, Latency ~10ms

**FAISS ANN (512-dim embeddings, 13k items):**
- Flat: ~1ms query, exact results
- IVF: ~0.5ms query, 98%+ recall
- HNSW: ~0.1ms query, 99%+ recall

## Usage Examples

### Evaluate Custom Recommender

```python
from evaluation.metrics import RecommendationEvaluator

# Your recommender
class MyRecommender:
    def recommend(self, query_id, top_n=10):
        # Your logic here
        return [(product_id, score), ...]

# Generate recommendations
recommended_lists = []
for query_id in test_queries:
    recs = my_recommender.recommend(query_id)
    recommended_lists.append([pid for pid, _ in recs])

# Evaluate
evaluator = RecommendationEvaluator()
results = evaluator.evaluate(recommended_lists, ground_truth)
```

### Benchmark FAISS

```python
from evaluation.ann_indexing import benchmark_index, print_benchmark_results

results = benchmark_index(
    embeddings=your_embeddings,
    product_ids=your_product_ids,
    index_types=['Flat', 'IVF', 'HNSW'],
    n_queries=100
)

print_benchmark_results(results)
```

### Compare Systems

```python
evaluator = RecommendationEvaluator()

systems = {
    'Random': random_recommender,
    'TF-IDF': tfidf_recommender,
    'Hybrid': hybrid_recommender
}

all_results = {}
for name, system in systems.items():
    # Generate recommendations...
    results = evaluator.evaluate(recs, ground_truth)
    all_results[name] = results

# Compare
evaluator.compare_systems(all_results, metric='NDCG@10')
```

## Dependencies

**Required:**
- pandas, numpy, scikit-learn

**Optional:**
- matplotlib, seaborn (for visualizations)
- faiss-cpu or faiss-gpu (for ANN search)

## Next Steps

1. **Run the notebook** to establish baseline performance
2. **Generate image embeddings** using CLIP or ViT
3. **Benchmark with real embeddings** instead of synthetic ones
4. **Implement hybrid approach** combining text + image
5. **Deploy and A/B test** in production

## See Also

- `../Documentation/EVALUATION.md` - Generated evaluation report
- `../evaluation_benchmark.ipynb` - Interactive evaluation
- `../hybrid_recommender_example.py` - Multimodal example
