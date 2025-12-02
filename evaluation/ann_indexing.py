"""
Approximate Nearest Neighbor (ANN) Indexing
============================================

Implements FAISS-based ANN search for fast similarity retrieval.
Supports both dense embeddings (image/text) with performance benchmarks.

Requires: pip install faiss-cpu (or faiss-gpu for GPU support)
"""

import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pickle

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not installed. Install with: pip install faiss-cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS-based ANN index for fast similarity search.
    
    Supports multiple index types:
    - Flat: Exact search (baseline)
    - IVF: Inverted file index (faster, approximate)
    - HNSW: Hierarchical NSW graph (very fast, approximate)
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = 'Flat',
        metric: str = 'L2',
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: 'Flat', 'IVF', or 'HNSW'
            metric: 'L2' (Euclidean) or 'IP' (Inner Product/Cosine)
            nlist: Number of clusters for IVF (ignored for other types)
            nprobe: Number of clusters to search (IVF only)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self.product_ids = []
        self.is_trained = False
        
        logger.info(f"Initializing FAISS {index_type} index (dim={dimension}, metric={metric})")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.metric == 'L2':
            metric_type = faiss.METRIC_L2
        elif self.metric == 'IP':
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        if self.index_type == 'Flat':
            # Exact search (brute force)
            index = faiss.IndexFlatL2(self.dimension) if self.metric == 'L2' else faiss.IndexFlatIP(self.dimension)
        
        elif self.index_type == 'IVF':
            # Inverted file index (requires training)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_type)
            index.nprobe = self.nprobe
        
        elif self.index_type == 'HNSW':
            # Hierarchical NSW graph
            index = faiss.IndexHNSWFlat(self.dimension, 32, metric_type)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def build(
        self,
        embeddings: np.ndarray,
        product_ids: List[str],
        normalize: bool = False
    ):
        """
        Build index from embeddings.
        
        Args:
            embeddings: Array of shape (n_items, dimension)
            product_ids: List of product IDs corresponding to embeddings
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}"
            )
        
        if len(product_ids) != embeddings.shape[0]:
            raise ValueError("Number of product IDs must match number of embeddings")
        
        # Store product IDs
        self.product_ids = product_ids
        
        # Normalize if needed (converts L2 distance to cosine similarity)
        if normalize:
            logger.info("L2-normalizing embeddings for cosine similarity")
            faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = self._create_index()
        
        # Train if needed (IVF requires training)
        if self.index_type == 'IVF':
            logger.info(f"Training IVF index with {len(embeddings)} vectors...")
            start = time.time()
            self.index.train(embeddings.astype(np.float32))
            train_time = time.time() - start
            logger.info(f"Training completed in {train_time:.2f}s")
            self.is_trained = True
        
        # Add vectors to index
        logger.info(f"Adding {len(embeddings)} vectors to index...")
        start = time.time()
        self.index.add(embeddings.astype(np.float32))
        add_time = time.time() - start
        logger.info(f"Index built in {add_time:.2f}s")
        
        logger.info(f"Index contains {self.index.ntotal} vectors")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        normalize: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            top_k: Number of results to return
            normalize: Whether to normalize query (if embeddings were normalized)
        
        Returns:
            List of (product_id, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize if needed
        if normalize:
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k
        )
        
        # Convert to product IDs with scores
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.product_ids):  # Valid index
                product_id = self.product_ids[idx]
                # Convert distance to similarity score (smaller distance = higher similarity)
                # For L2: similarity = 1 / (1 + distance)
                # For IP: distance is already similarity (higher is better)
                if self.metric == 'IP':
                    score = float(dist)
                else:
                    score = 1.0 / (1.0 + float(dist))
                results.append((product_id, score))
        
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        normalize: bool = False
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Array of shape (n_queries, dimension)
            top_k: Number of results per query
            normalize: Whether to normalize queries
        
        Returns:
            List of results for each query
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Normalize if needed
        if normalize:
            faiss.normalize_L2(query_embeddings)
        
        # Batch search
        distances, indices = self.index.search(
            query_embeddings.astype(np.float32),
            top_k
        )
        
        # Convert to results
        all_results = []
        for query_dists, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_dists, query_indices):
                if idx >= 0 and idx < len(self.product_ids):
                    product_id = self.product_ids[idx]
                    if self.metric == 'IP':
                        score = float(dist)
                    else:
                        score = 1.0 / (1.0 + float(dist))
                    results.append((product_id, score))
            all_results.append(results)
        
        return all_results
    
    def save(self, path: str):
        """Save index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = str(path.with_suffix('.faiss'))
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'product_ids': self.product_ids,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'nlist': self.nlist,
            'nprobe': self.nprobe
        }
        
        metadata_path = str(path.with_suffix('.pkl'))
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, path: str):
        """Load index and metadata from disk."""
        path = Path(path)
        
        # Load FAISS index
        index_path = str(path.with_suffix('.faiss'))
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = str(path.with_suffix('.pkl'))
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.product_ids = metadata['product_ids']
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self.metric = metadata['metric']
        self.nlist = metadata.get('nlist', 100)
        self.nprobe = metadata.get('nprobe', 10)
        
        logger.info(f"Index loaded from {index_path}")
        logger.info(f"Contains {self.index.ntotal} vectors")


def benchmark_index(
    embeddings: np.ndarray,
    product_ids: List[str],
    index_types: List[str] = ['Flat', 'IVF', 'HNSW'],
    top_k: int = 10,
    n_queries: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different FAISS index types.
    
    Args:
        embeddings: Array of embeddings (n_items, dimension)
        product_ids: List of product IDs
        index_types: List of index types to benchmark
        top_k: Number of neighbors to retrieve
        n_queries: Number of test queries
    
    Returns:
        Dictionary with benchmark results for each index type
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not installed")
    
    dimension = embeddings.shape[1]
    n_items = embeddings.shape[0]
    
    # Sample query embeddings
    query_indices = np.random.choice(n_items, size=min(n_queries, n_items), replace=False)
    query_embeddings = embeddings[query_indices]
    
    results = {}
    
    for index_type in index_types:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmarking {index_type} Index")
        logger.info(f"{'=' * 60}")
        
        # Create and build index
        index = FAISSIndex(
            dimension=dimension,
            index_type=index_type,
            metric='L2'
        )
        
        # Build time
        build_start = time.time()
        index.build(embeddings, product_ids, normalize=True)
        build_time = time.time() - build_start
        
        # Search time (single query)
        single_times = []
        for q in query_embeddings[:10]:  # Test on 10 queries
            search_start = time.time()
            _ = index.search(q, top_k=top_k, normalize=True)
            single_times.append(time.time() - search_start)
        avg_single_time = np.mean(single_times)
        
        # Batch search time
        batch_start = time.time()
        _ = index.batch_search(query_embeddings, top_k=top_k, normalize=True)
        batch_time = time.time() - batch_start
        avg_batch_time = batch_time / len(query_embeddings)
        
        # Store results
        results[index_type] = {
            'build_time_s': build_time,
            'single_query_ms': avg_single_time * 1000,
            'batch_query_ms': avg_batch_time * 1000,
            'throughput_qps': len(query_embeddings) / batch_time
        }
        
        logger.info(f"Build time: {build_time:.2f}s")
        logger.info(f"Single query: {avg_single_time*1000:.2f}ms")
        logger.info(f"Batch query: {avg_batch_time*1000:.2f}ms")
        logger.info(f"Throughput: {results[index_type]['throughput_qps']:.1f} QPS")
    
    return results


def print_benchmark_results(results: Dict[str, Dict[str, float]]):
    """Print formatted benchmark results."""
    print("\n" + "=" * 80)
    print("FAISS INDEX BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Index Type':<15} {'Build Time':<12} {'Single Query':<15} {'Batch Query':<15} {'Throughput':<12}")
    print("-" * 80)
    
    for index_type, metrics in results.items():
        print(
            f"{index_type:<15} "
            f"{metrics['build_time_s']:>10.2f}s "
            f"{metrics['single_query_ms']:>13.2f}ms "
            f"{metrics['batch_query_ms']:>13.2f}ms "
            f"{metrics['throughput_qps']:>10.1f} QPS"
        )
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    if not FAISS_AVAILABLE:
        print("❌ FAISS not installed. Install with: pip install faiss-cpu")
    else:
        print("Testing FAISS ANN Indexing\n")
        
        # Generate synthetic embeddings for testing
        n_items = 13000  # Simulate 13k products
        dimension = 512  # Common embedding dimension
        
        print(f"Generating {n_items} random {dimension}-dim embeddings...")
        embeddings = np.random.randn(n_items, dimension).astype(np.float32)
        product_ids = [f"product_{i:05d}" for i in range(n_items)]
        
        # Benchmark
        print("\nRunning benchmarks...\n")
        results = benchmark_index(
            embeddings,
            product_ids,
            index_types=['Flat', 'IVF', 'HNSW'],
            n_queries=100
        )
        
        # Print results
        print_benchmark_results(results)
        
        # Test save/load
        print("Testing save/load functionality...")
        index = FAISSIndex(dimension=dimension, index_type='Flat')
        index.build(embeddings, product_ids, normalize=True)
        
        save_path = "data/test_index"
        index.save(save_path)
        
        index2 = FAISSIndex(dimension=dimension, index_type='Flat')
        index2.load(save_path)
        
        print("✅ Save/load test passed\n")
