import pandas as pd
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import logging
from typing import Optional, List, Tuple
import hashlib
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image_url(url: str, timeout: int = 5) -> bool:
    """
    Validate if an image URL is accessible and returns valid image data.
    
    Args:
        url: Image URL to validate
        timeout: Request timeout in seconds
        
    Returns:
        True if URL is valid and returns an image, False otherwise
    """
    if not isinstance(url, str) or not url.strip():
        return False
    
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            return 'image' in content_type
    except Exception as e:
        logger.debug(f"Error validating URL {url}: {e}")
    
    return False

def download_image(url: str, timeout: int = 5) -> Optional[Image.Image]:
    """
    Download an image from a URL and return as PIL Image object.
    
    Args:
        url: Image URL to download
        timeout: Request timeout in seconds
        
    Returns:
        PIL Image object if successful, None otherwise
    """
    if not isinstance(url, str) or not url.strip():
        return None
    
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.verify()  # Verify it's a valid image
            # Re-open after verify() closes the file
            img = Image.open(BytesIO(response.content))
            return img
    except Exception as e:
        logger.debug(f"Error downloading image from {url}: {e}")
    
    return None

def cache_image_locally(image: Image.Image, cache_dir: str, product_id: str) -> Optional[str]:
    """
    Save image to local cache directory.
    
    Args:
        image: PIL Image object to cache
        cache_dir: Directory path to store cached images
        product_id: Product ID to use as filename
        
    Returns:
        Path to cached image if successful, None otherwise
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        image_path = Path(cache_dir) / f"{product_id}.jpg"
        image.convert('RGB').save(image_path, quality=95)
        return str(image_path)
    except Exception as e:
        logger.warning(f"Error caching image for {product_id}: {e}")
    
    return None

def process_image_urls(df: pd.DataFrame, cache_dir: str = "data/image_cache", 
                      validate_only: bool = False, download: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Process image URLs in the dataframe for multimodal embedding.
    
    Args:
        df: DataFrame with 'image_url' column
        cache_dir: Directory to cache downloaded images
        validate_only: If True, only validate URLs without downloading
        download: If True, download and cache valid images
        
    Returns:
        Tuple of (processed DataFrame, statistics dictionary)
    """
    stats = {
        'total_products': len(df),
        'valid_urls': 0,
        'invalid_urls': 0,
        'downloaded': 0,
        'cached': 0
    }
    
    if 'image_url' not in df.columns:
        logger.warning("'image_url' column not found in dataframe")
        return df, stats
    
    # Add columns for tracking
    df['image_url_valid'] = False
    df['image_local_path'] = None
    
    for idx, row in df.iterrows():
        url = row['image_url']
        
        if validate_image_url(url):
            stats['valid_urls'] += 1
            df.at[idx, 'image_url_valid'] = True
            
            if download:
                image = download_image(url)
                if image is not None:
                    stats['downloaded'] += 1
                    local_path = cache_image_locally(image, cache_dir, row['product_id'])
                    if local_path is not None:
                        stats['cached'] += 1
                        df.at[idx, 'image_local_path'] = local_path
        else:
            stats['invalid_urls'] += 1
    
    return df, stats

def preprocess_fashion_data(csv_path: str, process_images: bool = False, 
                           download_images: bool = False,
                           image_cache_dir: str = "data/image_cache") -> pd.DataFrame:
    """
    Preprocess fashion product data including optional image processing.
    
    Args:
        csv_path: Path to the CSV file
        process_images: If True, validate and process image URLs
        download_images: If True, download and cache valid images
        image_cache_dir: Directory for caching images
        
    Returns:
        Processed DataFrame
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"The file at {csv_path} does not exist.")

    # Load
    df = pd.read_csv(path)
    original = len(df)

    # Clean
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    cleaned = len(df)

    # Process images if requested
    if process_images:
        logger.info("Processing image URLs...")
        df, img_stats = process_image_urls(
            df, 
            cache_dir=image_cache_dir,
            validate_only=not download_images,
            download=download_images
        )
        logger.info(f"Image processing stats: {img_stats}")
    
    # Save (For testing)
    # output_path = path.with_name(path.stem + "_cleaned.csv")
    # df.to_csv(output_path, index=False)
    # print(f"Cleaned data saved to: {output_path}")

    # Print (For testing)
    print(f"Removed {original - cleaned} rows during preprocessing.")

    return df

if __name__ == "__main__":
    data_path = "../data/products.csv"
    
    # Example 1: Basic preprocessing without image processing
    print("=== Example 1: Basic Preprocessing ===")
    processed_df = preprocess_fashion_data(data_path)
    print(f"Processed dataframe shape: {processed_df.shape}\n")
    
    # Example 2: Validate image URLs only
    print("=== Example 2: Validate Image URLs ===")
    processed_df = preprocess_fashion_data(data_path, process_images=True, download_images=False)
    valid_count = processed_df['image_url_valid'].sum()
    print(f"Valid image URLs: {valid_count}/{len(processed_df)}\n")
    
    # Example 3: Download and cache images (uncomment to use)
    # print("=== Example 3: Download and Cache Images ===")
    # processed_df = preprocess_fashion_data(
    #     data_path, 
    #     process_images=True, 
    #     download_images=True,
    #     image_cache_dir="data/image_cache"
    # )
    # cached_count = processed_df['image_local_path'].notna().sum()
    # print(f"Cached images: {cached_count}/{len(processed_df)}")

