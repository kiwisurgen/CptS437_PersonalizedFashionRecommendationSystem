"""
Integration example: Combining text and image embeddings for recommendations.

This shows how to use the new image processing with existing TFIDF similarity.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Existing modules
from preprocessing.preprocess_product_data import preprocess_fashion_data
from processing.tfidf_title_similarity import tfidf_cosine_sim
from processing.onehot_generic_similarity import onehot_cosine_sim
from processing.price_similarity import price_similarity
from processing.rating_similarity import rating_similarity
from processing.image_embedding import ImageEmbeddingProcessor


class HybridRecommender:
    """
    Multimodal recommendation system combining text and image embeddings.
    """
    
    def __init__(self, csv_path: str, image_cache_dir: str = "data/image_cache"):
        """
        Initialize the hybrid recommender.
        
        Args:
            csv_path: Path to products CSV
            image_cache_dir: Directory for cached images
        """
        self.csv_path = csv_path
        self.image_processor = ImageEmbeddingProcessor(cache_dir=image_cache_dir)
        self.df = None
        self.tfidf_sims = None
        
    def prepare_data(self, process_images: bool = True, download_images: bool = False):
        """
        Prepare data: preprocess CSV and optionally download images.
        
        Args:
            process_images: Whether to validate image URLs
            download_images: Whether to download images to cache
        """
        print("Loading and preprocessing data...")
        self.df = preprocess_fashion_data(
            csv_path=self.csv_path,
            process_images=process_images,
            download_images=download_images,
            image_cache_dir=self.image_processor.cache_dir
        )
        print(f"Loaded {len(self.df)} products")
        
        if process_images:
            valid_urls = self.df['image_url_valid'].sum() if 'image_url_valid' in self.df.columns else 0
            print(f"Valid image URLs: {valid_urls}")
    
    def compute_text_similarities(self, product_idx: int, top_n: int = 5) -> list:
        """
        Compute text-based similarity using TF-IDF.
        
        Args:
            product_idx: Index of reference product
            top_n: Number of similar products to return
            
        Returns:
            List of (product_index, similarity_score) tuples
        """
        products = self.df['title'].tolist()
        sim_scores = tfidf_cosine_sim(idx=product_idx, n=top_n, products=products)
        return sim_scores
    
    def compute_category_similarities(self, product_idx: int, top_n: int = 5) -> list:
        categories = self.df['category'].tolist()
        sim_scores = onehot_cosine_sim(idx=product_idx, n=top_n, items=categories)
        return sim_scores
    
    def compute_brand_similarities(self, product_idx: int, top_n: int = 5) -> list: 
        brands = self.df['brand'].tolist()
        sim_scores = onehot_cosine_sim(idx=product_idx, n=top_n, items=brands)
        return sim_scores
    
    def compute_price_similarities(self, product_idx: int, top_n: int = 5) -> list:
        prices = self.df['price'].tolist()
        sim_scores = price_similarity(idx=product_idx, n=top_n, prices=prices, alpha=3.0, normalize=True)
        return sim_scores
    
    def compute_rating_similarities(self, product_idx: int, top_n: int = 5) -> list:
        ratings = self.df['rating'].tolist()
        sim_scores = rating_similarity(idx=product_idx, n=top_n, ratings=ratings, alpha=2.0, normalize=True)
        return sim_scores
    
    def compute_image_similarity(self, product_idx: int, top_n: int = 5, 
                                embedding_model=None) -> list:
        """
        Compute image-based similarity using cached embeddings.
        
        Args:
            product_idx: Index of reference product
            top_n: Number of similar products to return
            embedding_model: Model to generate embeddings (optional)
            
        Returns:
            List of (product_index, similarity_score) tuples
        """
        if embedding_model is None:
            print("Note: embedding_model not provided. Would need image embeddings.")
            return []
        
        # Load reference image
        product_id = self.df.iloc[product_idx]['product_id']
        ref_image = self.image_processor.load_cached_image(product_id)
        
        if ref_image is None:
            print(f"Could not load cached image for {product_id}")
            return []
        
        # Get embedding for reference product
        ref_embedding = embedding_model(np.expand_dims(ref_image, 0))
        
        # Compute similarities with all products
        similarities = []
        for idx, row in self.df.iterrows():
            if idx == product_idx:
                continue
            
            other_image = self.image_processor.load_cached_image(row['product_id'])
            if other_image is not None:
                other_embedding = embedding_model(np.expand_dims(other_image, 0))
                
                # Cosine similarity
                sim = np.dot(ref_embedding.flatten(), other_embedding.flatten()) / (
                    np.linalg.norm(ref_embedding) * np.linalg.norm(other_embedding) + 1e-8
                )
                similarities.append((idx, sim))
        
        # Sort and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def get_hybrid_recommendations(self, product_idx: int, top_n: int = 5,
                                   text_weight: float = 0.3,
                                   image_weight: float = 0.3,
                                   category_weight: float = 0.1,
                                   brand_weight: float = 0.1,
                                   price_weight: float = 0.1,
                                   rating_weight: float = 0.1,
                                   embedding_model=None) -> list:
        """
        Get recommendations using hybrid approach (text + image).
        
        Args:
            product_idx: Index of reference product
            top_n: Number of recommendations to return
            text_weight: Weight for text similarity (0-1)
            image_weight: Weight for image similarity (0-1)
            embedding_model: Image embedding model (optional)
            
        Returns:
            List of recommendations with hybrid scores
        """
        # Ensure weights sum to 1
        total_weight = text_weight + image_weight + category_weight + brand_weight + price_weight + rating_weight
        text_weight = text_weight / total_weight
        image_weight = image_weight / total_weight
        category_weight = category_weight / total_weight
        brand_weight = brand_weight / total_weight
        price_weight = price_weight / total_weight
        rating_weight = rating_weight / total_weight
        
        # Compute both similarities
        text_sims = self.compute_text_similarities(product_idx, top_n=top_n*2)
        image_sims = self.compute_image_similarity(product_idx, top_n=top_n*2, 
                                                   embedding_model=embedding_model)
        cat_sims = self.compute_category_similarities(product_idx, top_n=top_n*2)
        brand_sims = self.compute_brand_similarities(product_idx, top_n=top_n*2)
        price_sims = self.compute_price_similarities(product_idx, top_n=top_n*2)
        rating_sims = self.compute_rating_similarities(product_idx, top_n=top_n*2)
        
        # Create score dictionaries
        text_scores = {idx: score for idx, score in text_sims}
        image_scores = {idx: score for idx, score in image_sims}
        cat_scores = {idx: score for idx, score in cat_sims}
        brand_scores = {idx: score for idx, score in brand_sims}
        price_scores = {idx: score for idx, score in price_sims}
        rating_scores = {idx: score for idx, score in rating_sims}
        
        # Combine scores for all products
        all_product_ids = set(text_scores.keys()) | set(image_scores.keys()) | set(cat_scores.keys()) | \
                          set(brand_scores.keys()) | set(price_scores.keys()) | set(rating_scores.keys())
        hybrid_scores = []
        
        for idx in all_product_ids:
            text_sim = text_scores.get(idx, 0.0)
            image_sim = image_scores.get(idx, 0.0)
            cat_sim = cat_scores.get(idx, 0.0)
            brand_sim = brand_scores.get(idx, 0.0)
            price_sim = price_scores.get(idx, 0.0)
            rating_sim = rating_scores.get(idx, 0.0)
            hybrid_sim = (text_weight * text_sim) + (image_weight * image_sim) + \
                         (category_weight * cat_sim) + (brand_weight * brand_sim) + \
                         (price_weight * price_sim) + (rating_weight * rating_sim)
            hybrid_scores.append((idx, hybrid_sim))
        
        # Sort and return top N
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:top_n]
    
    def display_recommendations(self, recommendations: list, 
                               title: str = "Recommendations"):
        """
        Display recommendations in a readable format.
        
        Args:
            recommendations: List of (idx, score) tuples
            title: Title for the output
        """
        print(f"\n{title}")
        print("-" * 70)
        
        for rank, (idx, score) in enumerate(recommendations, 1):
            product = self.df.iloc[idx]
            print(f"{rank}. {product['title'][:60]}")
            print(f"   Brand: {product['brand']} | Price: ${product['price']} | Score: {score:.4f}")
        
        print()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("MULTIMODAL RECOMMENDATION EXAMPLE")
    print("=" * 70)
    print()
    
    # Initialize recommender
    recommender = HybridRecommender(csv_path="data/products.csv")
    
    # Step 1: Prepare data (validate but don't download for this example)
    print("Step 1: Preparing data...")
    recommender.prepare_data(process_images=True, download_images=False)
    
    # Step 2: Text-based recommendations
    print("\nStep 2: Text-based recommendations (using existing TFIDF)...")
    ref_product_idx = 5
    ref_product = recommender.df.iloc[ref_product_idx]
    print(f"Reference product: {ref_product['title']}")
    
    text_recs = recommender.compute_text_similarities(ref_product_idx, top_n=5)
    recommender.display_recommendations(text_recs, "Top 5 Similar by Title")
    
    # Step 3: Text + Image recommendations (without image embeddings for now)
    print("Step 3: Hybrid recommendations (text + image weight)...")
    print("Note: Image embeddings require embedding model. Using text only for demo.\n")
    
    hybrid_recs = recommender.get_hybrid_recommendations(
        product_idx=ref_product_idx,
        top_n=5,
        text_weight=0.3,
        image_weight=0.5,
        category_weight = 0.1,
        brand_weight = 0.1,
        price_weight = 0.1,
        rating_weight = 0.1,
        embedding_model=None  # Would need actual embedding model
    )
    recommender.display_recommendations(hybrid_recs, "Top 5 Hybrid Recommendations")
    
    # Step 4: Show integration pattern
    print("=" * 70)
    print("INTEGRATION PATTERN FOR FULL SYSTEM")
    print("=" * 70)
    print("""
To use with real image embeddings:

1. Import your embedding model:
   from transformers import CLIPModel, CLIPProcessor
   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

2. Download images (optional, for production):
   recommender.prepare_data(process_images=True, download_images=True)

3. Create embedding wrapper:
   def get_embedding(image_array):
       tensor = torch.from_numpy(image_array).unsqueeze(0)
       with torch.no_grad():
           return model.get_image_features(tensor).cpu().numpy()

4. Get hybrid recommendations:
   recs = recommender.get_hybrid_recommendations(
       product_idx=5,
       top_n=10,
       text_weight=0.4,
       image_weight=0.6,
       embedding_model=get_embedding
   )
    """)
