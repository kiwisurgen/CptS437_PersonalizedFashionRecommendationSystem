"""Quick test script to validate image processing pipeline.
Run this to verify the system works end-to-end.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.preprocess_product_data import preprocess_fashion_data, validate_image_url
from processing.image_embedding import ImageEmbeddingProcessor
import pandas as pd

# ========================================
# CONFIGURATION: Adjust sample sizes here
# ========================================
URL_VALIDATION_SAMPLE_SIZE = 20  # Number of products to validate URLs (TEST 2)
BATCH_PROCESSING_SAMPLE_SIZE = 3  # Number of images to download and process (TEST 4)


def test_url_validation():
    """Test URL validation functionality."""
    print("=" * 60)
    print("TEST 1: URL Validation")
    print("=" * 60)
    
    test_urls = [
        "https://m.media-amazon.com/images/I/51y2EF0OmOL._AC_UL320_.jpg",  # Valid
        "https://invalid-url-that-does-not-exist.com/image.jpg",  # Invalid
        None,  # Invalid (None)
        "",    # Invalid (empty)
    ]
    
    for url in test_urls:
        is_valid = validate_image_url(url, timeout=3)
        print(f"  URL: {str(url)[:50]}... | Valid: {is_valid}")
    
    print()


def test_image_url_processing():
    """Test image URL processing from CSV."""
    print("=" * 60)
    print("TEST 2: Image URL Processing from CSV")
    print("=" * 60)
    
    csv_path = "data/products.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found at {csv_path}")
        return
    
    print(f"  Loading CSV from {csv_path}...")
    print(f"  Testing with {URL_VALIDATION_SAMPLE_SIZE} products (for speed)...")
    
    # Load only a sample for testing
    df_full = pd.read_csv(csv_path)
    df_sample = df_full.head(URL_VALIDATION_SAMPLE_SIZE)
    
    # Save sample to temp file and process
    temp_csv = "data/temp_sample.csv"
    df_sample.to_csv(temp_csv, index=False)
    
    df = preprocess_fashion_data(
        csv_path=temp_csv,
        process_images=True,
        download_images=False  # Only validate, don't download
    )
    
    # Clean up temp file
    if Path(temp_csv).exists():
        Path(temp_csv).unlink()
    
    total = len(df)
    valid = df['image_url_valid'].sum()
    invalid = total - valid
    
    print(f"  Total products tested: {total}")
    print(f"  ✅ Valid image URLs: {valid} ({100*valid//total if total > 0 else 0}%)")
    print(f"  ❌ Invalid image URLs: {invalid} ({100*invalid//total if total > 0 else 0}%)")
    
    # Show sample of valid URLs
    if valid > 0:
        print(f"\n  Sample valid products:")
        valid_df = df[df['image_url_valid']].head(3)
        for idx, row in valid_df.iterrows():
            print(f"    - {row['product_id']}: {row['title'][:50]}")
    
    print()


def test_single_image_processing():
    """Test downloading and processing a single image."""
    print("=" * 60)
    print("TEST 3: Single Image Download & Processing")
    print("=" * 60)
    
    processor = ImageEmbeddingProcessor(
        cache_dir="data/image_cache",
        target_size=(224, 224)
    )
    
    # Get first valid URL from CSV
    csv_path = "data/products.csv"
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    sample_url = df.iloc[0]['image_url']
    product_id = df.iloc[0]['product_id']
    
    print(f"  Product ID: {product_id}")
    print(f"  URL: {sample_url}")
    print(f"  Processing...")
    
    result = processor.process_image_url(sample_url, product_id)
    
    if result:
        print(f"  ✅ Success!")
        print(f"    - Image shape: {result['shape']}")
        print(f"    - Cache path: {result['cache_path']}")
        print(f"    - Array dtype: {result['image_array'].dtype}")
        print(f"    - Value range: [{result['image_array'].min():.2f}, {result['image_array'].max():.2f}]")
    else:
        print(f"  ❌ Failed to process image")
    
    print()


def test_batch_processing():
    """Test batch image processing."""
    print("=" * 60)
    print("TEST 4: Batch Image Processing")
    print("=" * 60)
    
    csv_path = "data/products.csv"
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df_sample = df.head(BATCH_PROCESSING_SAMPLE_SIZE)  # Use configurable sample size
    
    processor = ImageEmbeddingProcessor(
        cache_dir="data/image_cache",
        target_size=(224, 224)
    )
    
    print(f"  Processing {len(df_sample)} products...")
    processed_df = processor.batch_process_images(df_sample)
    
    successful = processed_df['image_processed'].sum()
    print(f"  ✅ Successfully processed: {successful}/{len(df_sample)}")
    
    if successful > 0:
        print(f"  📁 Cache directory: {processor.cache_dir}")
        cached_files = list(Path(processor.cache_dir).glob("*.jpg"))
        print(f"  📊 Cached image files: {len(cached_files)}")
    
    print()


def test_load_cached():
    """Test loading cached images."""
    print("=" * 60)
    print("TEST 5: Load Cached Images")
    print("=" * 60)
    
    processor = ImageEmbeddingProcessor(cache_dir="data/image_cache")
    
    # Find cached images
    cache_dir = Path("data/image_cache")
    if not cache_dir.exists():
        print(f"  ℹ️  No cached images found (cache directory doesn't exist)")
        print()
        return
    
    cached_files = list(cache_dir.glob("*.jpg"))
    
    if not cached_files:
        print(f"  ℹ️  No cached images found")
    else:
        print(f"  Found {len(cached_files)} cached images")
        
        # Load first cached image
        first_file = cached_files[0]
        product_id = first_file.stem
        
        print(f"  Loading: {product_id}")
        image_array = processor.load_cached_image(product_id)
        
        if image_array is not None:
            print(f"  ✅ Successfully loaded")
            print(f"    - Shape: {image_array.shape}")
            print(f"    - Dtype: {image_array.dtype}")
            print(f"    - Range: [{image_array.min():.2f}, {image_array.max():.2f}]")
        else:
            print(f"  ❌ Failed to load")
    
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  IMAGE PROCESSING PIPELINE - VERIFICATION TESTS  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        test_url_validation()
        test_image_url_processing()
        test_single_image_processing()
        test_batch_processing()
        test_load_cached()
        
        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Review IMAGE_PROCESSING.md for full documentation")
        print("  2. Check image_cache/ for downloaded images")
        print("  3. Integrate with your embedding model")
        print()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
