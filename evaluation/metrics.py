"""
Evaluation Metrics for Recommendation System
=============================================

Implements standard information retrieval metrics:
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Hit Rate@K

Supports both binary relevance (0/1) and graded relevance (0-5 ratings).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Precision@K measures the proportion of recommended items in the top-K 
    that are relevant.
    
    Args:
        recommended: List of recommended item IDs (ordered by rank)
        relevant: List of relevant item IDs (ground truth)
        k: Number of top items to consider
    
    Returns:
        Precision@K score (0.0 to 1.0)
    
    Example:
        >>> recommended = ['A', 'B', 'C', 'D', 'E']
        >>> relevant = ['B', 'D', 'F']
        >>> precision_at_k(recommended, relevant, k=3)
        0.333  # 1 out of 3 (only 'B' is relevant)
    """
    if k <= 0:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Recall@K measures the proportion of relevant items that appear 
    in the top-K recommendations.
    
    Args:
        recommended: List of recommended item IDs (ordered by rank)
        relevant: List of relevant item IDs (ground truth)
        k: Number of top items to consider
    
    Returns:
        Recall@K score (0.0 to 1.0)
    
    Example:
        >>> recommended = ['A', 'B', 'C', 'D', 'E']
        >>> relevant = ['B', 'D', 'F']
        >>> recall_at_k(recommended, relevant, k=5)
        0.667  # 2 out of 3 relevant items found
    """
    if not relevant:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(relevant_set)


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    DCG emphasizes retrieving highly relevant documents early.
    Uses log2(i+2) discounting to penalize lower ranks more heavily.
    
    Args:
        relevance_scores: Relevance scores in recommendation order
        k: Number of top items to consider
    
    Returns:
        DCG@K score
    
    Formula:
        DCG@K = Σ(i=1 to k) (2^rel_i - 1) / log2(i + 1)
    """
    if k <= 0:
        return 0.0
    
    relevance_k = relevance_scores[:k]
    dcg = 0.0
    
    for i, rel in enumerate(relevance_k, start=1):
        dcg += (2**rel - 1) / np.log2(i + 1)
    
    return dcg


def ndcg_at_k(
    recommended: List[str],
    relevance_dict: Dict[str, float],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    NDCG normalizes DCG by the ideal DCG (if items were perfectly ranked).
    Returns a score between 0 and 1, where 1 is perfect ranking.
    
    Args:
        recommended: List of recommended item IDs (ordered by rank)
        relevance_dict: Dictionary mapping item IDs to relevance scores
        k: Number of top items to consider
    
    Returns:
        NDCG@K score (0.0 to 1.0)
    
    Example:
        >>> recommended = ['A', 'B', 'C']
        >>> relevance_dict = {'A': 3, 'B': 2, 'C': 0, 'D': 5}
        >>> ndcg_at_k(recommended, relevance_dict, k=3)
        0.789  # Not perfectly ranked (D should be first)
    """
    if k <= 0 or not relevance_dict:
        return 0.0
    
    # Get relevance scores for recommended items
    recommended_k = recommended[:k]
    relevance_scores = [relevance_dict.get(item, 0.0) for item in recommended_k]
    
    # Calculate DCG for recommended ranking
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate ideal DCG (sort by relevance, descending)
    all_relevances = sorted(relevance_dict.values(), reverse=True)
    ideal_dcg = dcg_at_k(all_relevances, k)
    
    # Normalize
    if ideal_dcg == 0.0:
        return 0.0
    
    return dcg / ideal_dcg


def mean_reciprocal_rank(
    recommended_lists: List[List[str]],
    relevant_items: List[List[str]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR measures how far down the first relevant item appears on average.
    Useful when you care about finding at least one relevant item quickly.
    
    Args:
        recommended_lists: List of recommendation lists for each query
        relevant_items: List of relevant items for each query
    
    Returns:
        MRR score (0.0 to 1.0)
    
    Example:
        >>> recommended_lists = [['A', 'B', 'C'], ['X', 'Y', 'Z']]
        >>> relevant_items = [['B', 'D'], ['Y']]
        >>> mean_reciprocal_rank(recommended_lists, relevant_items)
        0.75  # (1/2 + 1/2) / 2 = 0.75
    """
    if not recommended_lists or not relevant_items:
        return 0.0
    
    reciprocal_ranks = []
    
    for recommended, relevant in zip(recommended_lists, relevant_items):
        relevant_set = set(relevant)
        
        # Find rank of first relevant item (1-indexed)
        for rank, item in enumerate(recommended, start=1):
            if item in relevant_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant item found
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def hit_rate_at_k(
    recommended_lists: List[List[str]],
    relevant_items: List[List[str]],
    k: int
) -> float:
    """
    Calculate Hit Rate@K.
    
    Hit Rate@K is the proportion of queries for which at least one 
    relevant item appears in the top-K recommendations.
    
    Args:
        recommended_lists: List of recommendation lists for each query
        relevant_items: List of relevant items for each query
        k: Number of top items to consider
    
    Returns:
        Hit Rate@K (0.0 to 1.0)
    
    Example:
        >>> recommended_lists = [['A', 'B', 'C'], ['X', 'Y', 'Z']]
        >>> relevant_items = [['B'], ['M']]
        >>> hit_rate_at_k(recommended_lists, relevant_items, k=3)
        0.5  # Hit in first query, miss in second
    """
    if not recommended_lists or not relevant_items:
        return 0.0
    
    hits = 0
    
    for recommended, relevant in zip(recommended_lists, relevant_items):
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        if recommended_k & relevant_set:  # Intersection not empty
            hits += 1
    
    return hits / len(recommended_lists)


def average_precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Average Precision@K.
    
    AP@K averages precision at each relevant item position up to K.
    Rewards relevant items appearing earlier in the ranking.
    
    Args:
        recommended: List of recommended item IDs (ordered by rank)
        relevant: List of relevant item IDs (ground truth)
        k: Number of top items to consider
    
    Returns:
        AP@K score (0.0 to 1.0)
    """
    if not relevant or k <= 0:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    precisions = []
    num_hits = 0
    
    for i, item in enumerate(recommended_k, start=1):
        if item in relevant_set:
            num_hits += 1
            precisions.append(num_hits / i)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / min(len(relevant_set), k)


def mean_average_precision_at_k(
    recommended_lists: List[List[str]],
    relevant_items: List[List[str]],
    k: int
) -> float:
    """
    Calculate Mean Average Precision@K (MAP@K).
    
    MAP@K is the mean of AP@K across all queries.
    
    Args:
        recommended_lists: List of recommendation lists for each query
        relevant_items: List of relevant items for each query
        k: Number of top items to consider
    
    Returns:
        MAP@K score (0.0 to 1.0)
    """
    if not recommended_lists or not relevant_items:
        return 0.0
    
    aps = [
        average_precision_at_k(rec, rel, k)
        for rec, rel in zip(recommended_lists, relevant_items)
    ]
    
    return np.mean(aps)


class RecommendationEvaluator:
    """
    Comprehensive evaluator for recommendation systems.
    
    Computes multiple metrics at different K values and provides
    aggregated statistics across test queries.
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of K values to compute metrics for
        """
        self.k_values = k_values
        self.results = {}
    
    def evaluate(
        self,
        recommended_lists: List[List[str]],
        relevant_items: List[List[str]],
        relevance_scores: Optional[List[Dict[str, float]]] = None,
        verbose: bool = True
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate recommendations with multiple metrics.
        
        Args:
            recommended_lists: List of recommendation lists for each query
            relevant_items: List of relevant items for each query
            relevance_scores: Optional graded relevance scores for NDCG
            verbose: Whether to print results
        
        Returns:
            Dictionary of metric results: {metric_name: {k: score}}
        """
        results = {}
        
        for k in self.k_values:
            # Precision@K
            precisions = [
                precision_at_k(rec, rel, k)
                for rec, rel in zip(recommended_lists, relevant_items)
            ]
            results[f'Precision@{k}'] = np.mean(precisions)
            
            # Recall@K
            recalls = [
                recall_at_k(rec, rel, k)
                for rec, rel in zip(recommended_lists, relevant_items)
            ]
            results[f'Recall@{k}'] = np.mean(recalls)
            
            # Hit Rate@K
            results[f'HitRate@{k}'] = hit_rate_at_k(
                recommended_lists, relevant_items, k
            )
            
            # MAP@K
            results[f'MAP@{k}'] = mean_average_precision_at_k(
                recommended_lists, relevant_items, k
            )
            
            # NDCG@K (if relevance scores provided)
            if relevance_scores:
                ndcgs = [
                    ndcg_at_k(rec, rel_scores, k)
                    for rec, rel_scores in zip(recommended_lists, relevance_scores)
                ]
                results[f'NDCG@{k}'] = np.mean(ndcgs)
        
        # MRR (not K-specific)
        results['MRR'] = mean_reciprocal_rank(recommended_lists, relevant_items)
        
        self.results = results
        
        if verbose:
            self.print_results()
        
        return results
    
    def print_results(self):
        """Print formatted evaluation results."""
        print("\n" + "=" * 60)
        print("RECOMMENDATION EVALUATION RESULTS")
        print("=" * 60)
        
        for metric, value in sorted(self.results.items()):
            print(f"  {metric:20s}: {value:.4f}")
        
        print("=" * 60 + "\n")
    
    def compare_systems(
        self,
        system_results: Dict[str, Dict[str, float]],
        metric: str = 'NDCG@10'
    ):
        """
        Compare multiple recommendation systems.
        
        Args:
            system_results: Dict mapping system names to their results
            metric: Metric to use for comparison (default: NDCG@10)
        """
        print(f"\nSYSTEM COMPARISON ({metric})")
        print("=" * 60)
        
        sorted_systems = sorted(
            system_results.items(),
            key=lambda x: x[1].get(metric, 0.0),
            reverse=True
        )
        
        for rank, (system_name, results) in enumerate(sorted_systems, start=1):
            score = results.get(metric, 0.0)
            print(f"  {rank}. {system_name:30s}: {score:.4f}")
        
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Testing Evaluation Metrics\n")
    
    # Binary relevance example
    recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
    relevant = ['item2', 'item4', 'item6']
    
    print(f"Recommended: {recommended}")
    print(f"Relevant: {relevant}\n")
    
    p5 = precision_at_k(recommended, relevant, k=5)
    r5 = recall_at_k(recommended, relevant, k=5)
    
    print(f"Precision@5: {p5:.3f}")
    print(f"Recall@5: {r5:.3f}")
    
    # Graded relevance example
    print("\n" + "-" * 60)
    print("Graded Relevance Example\n")
    
    relevance_dict = {
        'item1': 0,
        'item2': 3,
        'item3': 1,
        'item4': 5,
        'item5': 2,
        'item6': 4
    }
    
    ndcg5 = ndcg_at_k(recommended, relevance_dict, k=5)
    print(f"NDCG@5: {ndcg5:.3f}")
    
    # Multiple queries example
    print("\n" + "-" * 60)
    print("Multiple Queries Example\n")
    
    evaluator = RecommendationEvaluator(k_values=[3, 5])
    
    recommended_lists = [
        ['A', 'B', 'C', 'D', 'E'],
        ['X', 'Y', 'Z', 'W', 'V']
    ]
    
    relevant_items = [
        ['B', 'D'],
        ['Y', 'M']
    ]
    
    evaluator.evaluate(recommended_lists, relevant_items)
