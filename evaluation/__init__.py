"""
Evaluation Module for Fashion Recommendation System
====================================================

This module provides comprehensive evaluation tools including:
- Metrics: P@K, Recall@K, NDCG, MRR, Hit Rate, MAP
- Baselines: Random, Popularity, TF-IDF, Category-based
- ANN Indexing: FAISS-based fast similarity search
- Benchmarking: Performance measurement and comparison

Usage:
    from evaluation.metrics import RecommendationEvaluator
    from evaluation.baselines import TFIDFRecommender, PopularityRecommender
    from evaluation.ann_indexing import FAISSIndex
"""

__version__ = "1.0.0"

# Import key classes for convenience
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    RecommendationEvaluator
)

from evaluation.baselines import (
    RandomRecommender,
    PopularityRecommender,
    TFIDFRecommender,
    CategoryBasedRecommender
)

try:
    from evaluation.ann_indexing import FAISSIndex, benchmark_index
    __all__ = [
        'precision_at_k', 'recall_at_k', 'ndcg_at_k',
        'mean_reciprocal_rank', 'hit_rate_at_k',
        'RecommendationEvaluator',
        'RandomRecommender', 'PopularityRecommender',
        'TFIDFRecommender', 'CategoryBasedRecommender',
        'FAISSIndex', 'benchmark_index'
    ]
except ImportError:
    __all__ = [
        'precision_at_k', 'recall_at_k', 'ndcg_at_k',
        'mean_reciprocal_rank', 'hit_rate_at_k',
        'RecommendationEvaluator',
        'RandomRecommender', 'PopularityRecommender',
        'TFIDFRecommender', 'CategoryBasedRecommender'
    ]
