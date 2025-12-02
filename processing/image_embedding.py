"""
Image embedding module for multimodal fashion recommendation system.

This module handles downloading images, extracting embeddings, and preparing
image features for the recommendation pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging
from PIL import Image
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class ImageEmbeddingProcessor:
    """Process images from URLs and prepare them for embedding models."""
    
    def __init__(self, cache_dir: str = "data/image_cache", 
                 target_size: tuple = (224, 224)):
        """
        Initialize the image processor.
        
        Args:
            cache_dir: Directory to store cached images
            target_size: Target image size for model input (height, width)
        """
        self.cache_dir = Path(cache_dir)
        self.target_size = target_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_image_from_url(self, url: str, timeout: int = 5) -> Optional[Image.Image]:
        """
        Download and return a PIL Image from URL.
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image object or None if download fails
        """
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img
        except Exception as e:
            logger.debug(f"Failed to download image from {url}: {e}")
        
        return None
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for embedding models.
        - Convert to RGB
        - Resize to target size
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image object
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size while maintaining aspect ratio
        image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        
        # Create a new image with target size and paste the resized image
        new_image = Image.new('RGB', self.target_size, (255, 255, 255))
        offset = ((self.target_size[0] - image.size[0]) // 2,
                  (self.target_size[1] - image.size[1]) // 2)
        new_image.paste(image, offset)
        
        return new_image
    
    def image_to_array(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to normalized numpy array.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized numpy array [0, 1] with shape (height, width, 3)
        """
        arr = np.array(image, dtype=np.float32) / 255.0
        return arr
    
    def process_image_url(self, url: str, product_id: str) -> Optional[Dict]:
        """
        Complete pipeline: download, preprocess, and prepare image for embedding.
        
        Args:
            url: Image URL
            product_id: Product ID for caching
            
        Returns:
            Dictionary with image data or None if processing fails
        """
        # Download
        image = self.download_image_from_url(url)
        if image is None:
            return None
        
        # Preprocess
        image = self.preprocess_image(image)
        
        # Cache locally
        cache_path = self.cache_dir / f"{product_id}.jpg"
        try:
            image.save(cache_path, quality=95)
        except Exception as e:
            logger.warning(f"Failed to cache image for {product_id}: {e}")
            cache_path = None
        
        # Convert to array
        image_array = self.image_to_array(image)
        
        return {
            'image_array': image_array,
            'cache_path': str(cache_path) if cache_path else None,
            'shape': image_array.shape,
            'url': url
        }
    
    def batch_process_images(self, df: pd.DataFrame, 
                            url_column: str = 'image_url',
                            id_column: str = 'product_id',
                            skip_existing: bool = True) -> pd.DataFrame:
        """
        Process images for multiple products.
        
        Args:
            df: DataFrame with image URLs
            url_column: Column name containing image URLs
            id_column: Column name containing product IDs
            skip_existing: Skip processing if cache file already exists
            
        Returns:
            DataFrame with added image processing results
        """
        results = []
        
        for idx, row in df.iterrows():
            url = row[url_column]
            product_id = row[id_column]
            
            # Skip if cache exists and skip_existing is True
            cache_path = self.cache_dir / f"{product_id}.jpg"
            if skip_existing and cache_path.exists():
                results.append({
                    'image_processed': True,
                    'image_cache_path': str(cache_path),
                    'image_shape': '(224, 224, 3)'
                })
                continue
            
            # Process image
            result = self.process_image_url(url, product_id)
            
            if result is not None:
                results.append({
                    'image_processed': True,
                    'image_cache_path': result['cache_path'],
                    'image_shape': str(result['shape'])
                })
            else:
                results.append({
                    'image_processed': False,
                    'image_cache_path': None,
                    'image_shape': None
                })
        
        # Add results to dataframe
        results_df = pd.DataFrame(results)
        df = pd.concat([df, results_df], axis=1)
        
        return df
    
    def load_cached_image(self, product_id: str) -> Optional[np.ndarray]:
        """
        Load a cached image and convert to array.
        
        Args:
            product_id: Product ID to load
            
        Returns:
            Image as normalized numpy array or None
        """
        cache_path = self.cache_dir / f"{product_id}.jpg"
        
        if not cache_path.exists():
            return None
        
        try:
            image = Image.open(cache_path)
            return self.image_to_array(image)
        except Exception as e:
            logger.error(f"Failed to load cached image for {product_id}: {e}")
        
        return None
    
    def get_batch_embeddings(self, product_ids: List[str], 
                            embedding_model) -> Optional[np.ndarray]:
        """
        Load batch of cached images and generate embeddings.
        
        Args:
            product_ids: List of product IDs to get embeddings for
            embedding_model: Model with __call__ method to generate embeddings
                           Expected signature: model(image_array) -> embedding
            
        Returns:
            Array of shape (batch_size, embedding_dim) or None if fails
        """
        embeddings = []
        
        for product_id in product_ids:
            image_array = self.load_cached_image(product_id)
            
            if image_array is None:
                logger.warning(f"Could not load image for {product_id}")
                continue
            
            try:
                # Add batch dimension if needed
                if len(image_array.shape) == 3:
                    image_batch = np.expand_dims(image_array, axis=0)
                else:
                    image_batch = image_array
                
                embedding = embedding_model(image_batch)
                embeddings.append(embedding.flatten())
            except Exception as e:
                logger.error(f"Failed to generate embedding for {product_id}: {e}")
                continue
        
        if not embeddings:
            return None
        
        return np.array(embeddings)


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = ImageEmbeddingProcessor(cache_dir="data/image_cache")
    
    # Example 1: Process single image
    print("=== Example 1: Process Single Image ===")
    url = "https://m.media-amazon.com/images/I/51y2EF0OmOL._AC_UL320_.jpg"
    result = processor.process_image_url(url, "example_product_001")
    if result:
        print(f"Successfully processed image. Shape: {result['shape']}")
        print(f"Cached at: {result['cache_path']}")
    
    # Example 2: Batch process from dataframe
    # print("\n=== Example 2: Batch Process from DataFrame ===")
    # df = pd.read_csv("data/products.csv")
    # processed_df = processor.batch_process_images(df)
    # print(f"Successfully processed {processed_df['image_processed'].sum()} images")
