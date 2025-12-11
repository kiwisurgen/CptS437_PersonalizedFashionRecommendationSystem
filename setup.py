#!/usr/bin/env python
"""
Setup and initialization script for the Fashion Recommendation System.
Run this to initialize the project and verify everything is working.
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")


def print_success(text):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print_success(description)
            return True
        else:
            print_error(f"{description}: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"{description}: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'requests': 'requests',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print_success(f"{package_name} installed")
        except ImportError:
            print_error(f"{package_name} NOT installed")
            missing.append(package_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nTo install, run:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print_success("All required dependencies installed!")
    return True


def create_directories():
    """Create necessary project directories."""
    print_header("Creating Project Directories")
    
    directories = [
        "data",
        "data/image_cache",
        "preprocessing",
        "processing",
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print_info(f"Directory exists: {directory}/")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created directory: {directory}/")


def verify_data_files():
    """Verify that data files exist."""
    print_header("Verifying Data Files")
    
    files_to_check = [
        ("data/products.csv", "Product catalog"),
        ("preprocessing/preprocess_product_data.py", "Preprocessing module"),
        ("processing/tfidf_title_similarity.py", "TF-IDF similarity module"),
        ("processing/image_embedding.py", "Image embedding module"),
    ]
    
    all_exist = True
    for file_path, description in files_to_check:
        if Path(file_path).exists():
            print_success(f"Found: {description} ({file_path})")
        else:
            print_error(f"Missing: {description} ({file_path})")
            all_exist = False
    
    return all_exist


def run_tests():
    """Run verification tests."""
    print_header("Running Verification Tests")
    
    if not Path("test_image_pipeline.py").exists():
        print_error("Test file not found: test_image_pipeline.py")
        return False
    
    # Run tests with Python
    cmd = f"{sys.executable} test_image_pipeline.py"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print_error(f"Failed to run tests: {e}")
        return False


def display_next_steps():
    """Display next steps for users."""
    print_header("Next Steps")
    
    print("""
1. VALIDATE IMAGE URLS
   python -c "
from preprocessing.preprocess_product_data import preprocess_fashion_data
df = preprocess_fashion_data('data/products.csv', process_images=True, download_images=False)
print(f'Valid URLs: {df[\"image_url_valid\"].sum()}')
   "

2. DOWNLOAD IMAGES (Optional - for production)
   python -c "
from preprocessing.preprocess_product_data import preprocess_fashion_data
df = preprocess_fashion_data('data/products.csv', process_images=True, download_images=True)
   "

3. GET TEXT-BASED RECOMMENDATIONS
   python -c "
from preprocessing.preprocess_product_data import preprocess_fashion_data
from processing.tfidf_title_similarity import tfidf_cosine_sim
df = preprocess_fashion_data('data/products.csv')
products = df['title'].tolist()
sims = tfidf_cosine_sim(idx=5, n=5, products=products)
for idx, score in sims:
    print(f'{products[idx]}: {score:.4f}')
   "

4. USE HYBRID RECOMMENDER
   python hybrid_recommender_example.py

5. INTEGRATE WITH EMBEDDINGS
   See IMAGE_PROCESSING.md and INTEGRATION_SUMMARY.md for details

📚 Documentation:
   - README.md - Project overview
   - IMAGE_PROCESSING.md - Complete image processing guide
   - INTEGRATION_SUMMARY.md - Implementation overview
   - hybrid_recommender_example.py - Working example
    """)


def main():
    """Main setup script."""
    print_header("Fashion Recommendation System - Setup")
    
    steps = [
        ("Checking Dependencies", check_dependencies),
        ("Creating Directories", create_directories),
        ("Verifying Data Files", verify_data_files),
    ]
    
    all_passed = True
    for step_name, step_func in steps:
        try:
            if not step_func():
                all_passed = False
        except Exception as e:
            print_error(f"Error in {step_name}: {e}")
            all_passed = False
    
    if not all_passed:
        print_header("⚠️  Setup Issues Detected")
        print("""
Some checks failed. Please:
1. Install missing dependencies: pip install -r requirements.txt
2. Verify data files are present
3. Check file permissions
        """)
        return 1
    
    # Ask about running tests
    print_header("Ready for Testing")
    print("Would you like to run verification tests? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        run_tests()
    
    # Show next steps
    display_next_steps()
    
    print_header("✅ Setup Complete!")
    print("""
Your Fashion Recommendation System is ready to use!

Quick start:
  python hybrid_recommender_example.py

For more information:
  - See README.md for overview
  - See IMAGE_PROCESSING.md for detailed guides
  - Run: python test_image_pipeline.py for verification
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
