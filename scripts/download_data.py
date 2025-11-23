#!/usr/bin/env python3
"""
Download dataset from Kaggle for deforestation monitoring
"""

import os
import sys
from pathlib import Path
from zipfile import ZipFile
import kaggle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def download_dataset(dataset_name=None, output_dir=None):
    """
    Download dataset from Kaggle
    
    Args:
        dataset_name: Kaggle dataset identifier (default from config)
        output_dir: Output directory (default from config)
    """
    if dataset_name is None:
        dataset_name = config.KAGGLE_DATASET
    
    if output_dir is None:
        output_dir = config.RAW_DATA_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("KAGGLE DATASET DOWNLOADER")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    try:
        print("\n📥 Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True,
            quiet=False
        )
        print("\n✅ Download complete!")
        
        # List downloaded files
        print("\n📁 Downloaded files:")
        for item in output_dir.iterdir():
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  - {item.name} ({size_mb:.2f} MB)")
            elif item.is_dir():
                file_count = len(list(item.rglob('*')))
                print(f"  - {item.name}/ ({file_count} items)")
        
        print("\n✅ Dataset ready for use!")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("2. Verify the dataset name is correct")
        print("3. Ensure you have accepted the dataset's terms on Kaggle website")
        sys.exit(1)

def list_available_datasets():
    """List popular deforestation datasets on Kaggle"""
    print("\n📚 Popular Deforestation Datasets on Kaggle:")
    print("="*60)
    
    datasets = [
        "mbogernetto/brazilian-amazon-rainforest-degradation",
        "nikitarom/amazon-rainforest-satellite-images",
        "c-f-h/satellite-imagery-of-amazon",
    ]
    
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kaggle dataset for deforestation monitoring")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Kaggle dataset identifier (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory path"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List popular deforestation datasets"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
    else:
        download_dataset(args.dataset, args.output)
