"""
Load Winoground dataset from local files (for manual downloads)
Works with both JSONL and Parquet formats
"""

import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List
import zipfile

class WinogroundLocalLoader:
    """Load Winoground dataset from manually downloaded files"""
    
    def __init__(self, data_dir="data/winoground"):
        self.data_dir = Path(data_dir)
        self.examples = []
        self.images_dir = None
        
    def load(self):
        """Load dataset from local files"""
        print("=" * 60)
        print("LOADING WINOGROUND FROM LOCAL FILES")
        print("=" * 60)
        
        # Check if data directory exists
        if not self.data_dir.exists():
            print(f"❌ Directory not found: {self.data_dir}")
            print("Please download dataset manually")
            return False
        
        print(f"\n📁 Looking in: {self.data_dir}")
        
        # Try loading examples
        examples_loaded = self._load_examples()
        if not examples_loaded:
            return False
        
        # Try loading images
        images_loaded = self._load_images()
        
        if images_loaded:
            print(f"\n✅ Successfully loaded Winoground dataset")
            print(f"   Examples: {len(self.examples)}")
            print(f"   Images dir: {self.images_dir}")
            self._print_sample()
            return True
        else:
            print(f"\n⚠️ Loaded examples but images not found")
            print(f"   You can still work with text data")
            return True
    
    def _load_examples(self):
        """Load examples from JSONL or Parquet file"""
        
        # Try JSONL first
        jsonl_file = self.data_dir / "examples.jsonl"
        if jsonl_file.exists():
            print(f"✓ Found JSONL file: {jsonl_file}")
            try:
                with open(jsonl_file, 'r') as f:
                    self.examples = [json.loads(line) for line in f]
                print(f"✓ Loaded {len(self.examples)} examples from JSONL")
                return True
            except Exception as e:
                print(f"✗ Error loading JSONL: {e}")
        
        print(f"\n❌ No data files found!")
        return False
    
    def _load_images(self):
        """Check for images directory or zip file"""
        
        # Check for images directory
        images_dir = self.data_dir / "images"
        if images_dir.exists() and images_dir.is_dir():
            print(f"✓ Found images directory: {images_dir}")
            self.images_dir = images_dir
            
            # Count images
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            print(f"✓ Found {len(image_files)} image files")
            return True
        
        print(f"\n⚠️ Images not found")
        return False
    
    def _print_sample(self):
        """Print a sample example"""
        if not self.examples:
            return
        
        print(f"\n📝 Sample Example:")
        sample = self.examples[0]
        print(f"   ID: {sample.get('id', 'N/A')}")
        print(f"   Caption 0: {sample.get('caption_0', 'N/A')}")
        print(f"   Caption 1: {sample.get('caption_1', 'N/A')}")
        if 'tag' in sample:
            print(f"   Tag: {sample.get('tag', 'N/A')}")
    
    def get_example(self, idx):
        """Get a single example with images loaded"""
        if idx >= len(self.examples):
            return None
        
        example = self.examples[idx].copy()
        
        # Load images if available
        if self.images_dir:
            example_id = example.get('id', idx)
            img0_path = self.images_dir / f"ex_{example_id}_img_0.png"
            img1_path = self.images_dir / f"ex_{example_id}_img_1.png"
            
            if img0_path.exists():
                example['image_0'] = Image.open(img0_path)
            if img1_path.exists():
                example['image_1'] = Image.open(img1_path)
        
        return example
    
    def __len__(self):
        return len(self.examples)
    
    def __iter__(self):
        for i in range(len(self.examples)):
            yield self.get_example(i)