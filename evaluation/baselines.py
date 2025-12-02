"""
Baseline Recommender Systems
=============================

Implements simple baseline recommenders for comparison:
- Random: Random recommendations
- Popularity: Most popular items
- TF-IDF: Text-based similarity (existing implementation)
- Co-occurrence: Items bought/viewed together

These serve as baselines to evaluate more sophisticated models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.tfidf_title_similarity import tfidf_cosine_sim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineRecommender:
    """Base class for all recommenders."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize recommender with product data.
        
        Args:
            df: DataFrame with product information
        """
        self.df = df
        self.product_ids = df['product_id'].tolist()
        self.n_products = len(df)
    
    def recommend(
        self,
        query_product_id: str,
        top_n: int = 10,
        exclude_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations for a product.
        
        Args:
            query_product_id: Product to get recommendations for
            top_n: Number of recommendations
            exclude_query: Whether to exclude the query product
        
        Returns:
            List of (product_id, score) tuples
        """
        raise NotImplementedError("Subclasses must implement recommend()")
    
    def get_product_index(self, product_id: str) -> int:
        """Get DataFrame index for a product ID."""
        try:
            return self.df[self.df['product_id'] == product_id].index[0]
        except IndexError:
            raise ValueError(f"Product ID {product_id} not found")


class RandomRecommender(BaselineRecommender):
    """
    Random Recommender Baseline.
    
    Recommends random products. Useful as a sanity check - 
    any real recommender should beat this.
    """
    
    def __init__(self, df: pd.DataFrame, seed: int = 42):
        """
        Initialize random recommender.
        
        Args:
            df: DataFrame with product information
            seed: Random seed for reproducibility
        """
        super().__init__(df)
        self.rng = np.random.RandomState(seed)
        logger.info("Initialized RandomRecommender")
    
    def recommend(
        self,
        query_product_id: str,
        top_n: int = 10,
        exclude_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Return random products.
        
        Scores are random uniform [0, 1] values.
        """
        available_products = self.product_ids.copy()
        
        if exclude_query and query_product_id in available_products:
            available_products.remove(query_product_id)
        
        # Sample without replacement
        n_sample = min(top_n, len(available_products))
        selected = self.rng.choice(available_products, size=n_sample, replace=False)
        
        # Random scores
        scores = self.rng.uniform(0, 1, size=n_sample)
        
        # Sort by score descending
        recommendations = list(zip(selected, scores))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations


class PopularityRecommender(BaselineRecommender):
    """
    Popularity-Based Recommender.
    
    Recommends most popular items (by rating count, average rating, 
    or custom popularity metric). Simple but often effective baseline.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        popularity_metric: str = 'rating',
        min_interactions: int = 0
    ):
        """
        Initialize popularity recommender.
        
        Args:
            df: DataFrame with product information
            popularity_metric: Column to use for popularity ('rating', custom)
            min_interactions: Minimum interactions to consider
        """
        super().__init__(df)
        self.popularity_metric = popularity_metric
        self.min_interactions = min_interactions
        
        # Calculate popularity scores
        self._compute_popularity()
        logger.info(f"Initialized PopularityRecommender (metric: {popularity_metric})")
    
    def _compute_popularity(self):
        """Compute popularity scores for all products."""
        if self.popularity_metric == 'rating':
            # Use product rating as popularity
            self.popularity = self.df.set_index('product_id')['rating'].fillna(0.0).to_dict()
        
        elif self.popularity_metric == 'random':
            # Random scores (for testing)
            rng = np.random.RandomState(42)
            self.popularity = {
                pid: rng.uniform(0, 5) 
                for pid in self.product_ids
            }
        
        else:
            # Try to use custom column
            if self.popularity_metric in self.df.columns:
                self.popularity = self.df.set_index('product_id')[
                    self.popularity_metric
                ].fillna(0.0).to_dict()
            else:
                logger.warning(f"Unknown metric '{self.popularity_metric}', using rating")
                self.popularity = self.df.set_index('product_id')['rating'].fillna(0.0).to_dict()
    
    def recommend(
        self,
        query_product_id: str,
        top_n: int = 10,
        exclude_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Return most popular products.
        
        Ignores the query product (non-personalized).
        """
        # Sort all products by popularity
        sorted_products = sorted(
            self.popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Exclude query if needed
        if exclude_query:
            sorted_products = [
                (pid, score) for pid, score in sorted_products
                if pid != query_product_id
            ]
        
        return sorted_products[:top_n]


class TFIDFRecommender(BaselineRecommender):
    """
    TF-IDF Text Similarity Recommender.
    
    Uses existing TF-IDF implementation for text-based similarity.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize TF-IDF recommender.
        
        Args:
            df: DataFrame with product information (must have 'title' column)
        """
        super().__init__(df)
        
        if 'title' not in df.columns:
            raise ValueError("DataFrame must have 'title' column for TF-IDF")
        
        # Fill NaN titles with empty strings to avoid TF-IDF errors
        self.titles = df['title'].fillna('').astype(str).tolist()
        logger.info("Initialized TFIDFRecommender")
    
    def recommend(
        self,
        query_product_id: str,
        top_n: int = 10,
        exclude_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Return products with similar titles.
        
        Uses TF-IDF cosine similarity on product titles.
        """
        # Get index of query product
        query_idx = self.get_product_index(query_product_id)
        
        # Get similar products using existing TF-IDF function
        n = top_n + 1 if exclude_query else top_n
        similar = tfidf_cosine_sim(idx=query_idx, n=n, products=self.titles)
        
        # Convert to product IDs with scores
        recommendations = []
        for idx, score in similar:
            pid = self.product_ids[idx]
            
            if exclude_query and pid == query_product_id:
                continue
            
            recommendations.append((pid, score))
        
        return recommendations[:top_n]


class CategoryBasedRecommender(BaselineRecommender):
    """
    Category-Based Recommender.
    
    Recommends popular products from the same category.
    Combines content filtering (category) with popularity.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        category_column: str = 'category',
        popularity_metric: str = 'rating'
    ):
        """
        Initialize category-based recommender.
        
        Args:
            df: DataFrame with product information
            category_column: Column name for product category
            popularity_metric: Column to use for popularity within category
        """
        super().__init__(df)
        self.category_column = category_column
        self.popularity_metric = popularity_metric
        
        if category_column not in df.columns:
            raise ValueError(f"DataFrame must have '{category_column}' column")
        
        logger.info("Initialized CategoryBasedRecommender")
    
    def recommend(
        self,
        query_product_id: str,
        top_n: int = 10,
        exclude_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Return popular products from same category.
        """
        # Get query product's category
        query_row = self.df[self.df['product_id'] == query_product_id]
        
        if query_row.empty:
            raise ValueError(f"Product ID {query_product_id} not found")
        
        query_category = query_row.iloc[0][self.category_column]
        
        # Get all products in same category
        same_category = self.df[self.df[self.category_column] == query_category]
        
        if exclude_query:
            same_category = same_category[
                same_category['product_id'] != query_product_id
            ]
        
        # Sort by popularity metric
        if self.popularity_metric in same_category.columns:
            same_category = same_category.sort_values(
                by=self.popularity_metric,
                ascending=False
            )
        
        # Get top N
        top_products = same_category.head(top_n)
        
        # Return with scores (use popularity metric as score)
        recommendations = []
        for _, row in top_products.iterrows():
            pid = row['product_id']
            score = row.get(self.popularity_metric, 0.0)
            recommendations.append((pid, float(score)))
        
        return recommendations


def evaluate_baseline(
    recommender: BaselineRecommender,
    test_queries: List[str],
    ground_truth: Dict[str, List[str]],
    top_k: int = 10
) -> Dict[str, float]:
    """
    Evaluate a baseline recommender.
    
    Args:
        recommender: Baseline recommender instance
        test_queries: List of product IDs to test
        ground_truth: Dict mapping query IDs to relevant items
        top_k: Number of recommendations to evaluate
    
    Returns:
        Dictionary of metric scores
    """
    from evaluation.metrics import RecommendationEvaluator
    
    # Generate recommendations for all test queries
    recommended_lists = []
    relevant_lists = []
    
    for query_id in test_queries:
        try:
            # Get recommendations
            recs = recommender.recommend(query_id, top_n=top_k)
            recommended_ids = [pid for pid, score in recs]
            
            # Get ground truth
            relevant = ground_truth.get(query_id, [])
            
            recommended_lists.append(recommended_ids)
            relevant_lists.append(relevant)
        
        except Exception as e:
            logger.warning(f"Failed to get recommendations for {query_id}: {e}")
            continue
    
    # Evaluate
    evaluator = RecommendationEvaluator(k_values=[5, 10, 20])
    results = evaluator.evaluate(
        recommended_lists,
        relevant_lists,
        verbose=False
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing Baseline Recommenders\n")
    
    # Load sample data
    csv_path = "data/products.csv"
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} products\n")
        
        # Test each baseline
        query_product = df.iloc[100]['product_id']
        print(f"Query Product: {query_product}")
        print(f"Title: {df.iloc[100]['title']}\n")
        
        # Random
        print("-" * 60)
        print("Random Recommender:")
        random_rec = RandomRecommender(df)
        recs = random_rec.recommend(query_product, top_n=5)
        for i, (pid, score) in enumerate(recs, 1):
            title = df[df['product_id'] == pid].iloc[0]['title']
            print(f"  {i}. {title[:50]}... (score: {score:.3f})")
        
        # Popularity
        print("\n" + "-" * 60)
        print("Popularity Recommender:")
        pop_rec = PopularityRecommender(df)
        recs = pop_rec.recommend(query_product, top_n=5)
        for i, (pid, score) in enumerate(recs, 1):
            title = df[df['product_id'] == pid].iloc[0]['title']
            print(f"  {i}. {title[:50]}... (score: {score:.3f})")
        
        # TF-IDF
        print("\n" + "-" * 60)
        print("TF-IDF Recommender:")
        tfidf_rec = TFIDFRecommender(df)
        recs = tfidf_rec.recommend(query_product, top_n=5)
        for i, (pid, score) in enumerate(recs, 1):
            title = df[df['product_id'] == pid].iloc[0]['title']
            print(f"  {i}. {title[:50]}... (score: {score:.3f})")
    
    else:
        print(f"❌ CSV file not found at {csv_path}")
