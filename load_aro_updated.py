"""
Load ARO dataset from your downloaded files
Updated to match your actual file structure
"""

import os
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List

class AROLoader:
    """Load ARO dataset from local files"""
    
    def __init__(self, data_dir="data/aro"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
        self.file_mapping = {
            "vg_relation": "visual_genome_relation.json",
            "vg_attribution": "visual_genome_attribution.json"
        }
        
        self.images_dir = self.data_dir / "images"
    
    def load(self):
        """Load ARO datasets"""
        print("="*60)
        print("LOADING ARO DATASET")
        print("="*60)
        
        if not self.data_dir.exists():
            print(f"❌ Directory not found: {self.data_dir}")
            return False
        
        print(f"\n📁 Loading from: {self.data_dir}")
        
        loaded = 0
        for subset_name, filename in self.file_mapping.items():
            if self._load_subset(subset_name, filename):
                loaded += 1
        
        if loaded > 0:
            print(f"\n✅ Successfully loaded {loaded} ARO subsets")
            return True
        else:
            return False
    
    def _load_subset(self, subset_name, filename):
        """Load a single subset"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  {subset_name}: {filename} not found")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                self.datasets[subset_name] = data
            elif isinstance(data, dict) and 'data' in data:
                self.datasets[subset_name] = data['data']
            else:
                self.datasets[subset_name] = data
            
            print(f"✓ {subset_name:20s}: {len(self.datasets[subset_name]):,} examples")
            return True
            
        except Exception as e:
            print(f"✗ {subset_name}: error - {e}")
            return False
    
    def get_subset(self, subset_name):
        """Get a specific subset"""
        return self.datasets.get(subset_name, [])
    
    def __len__(self):
        return sum(len(data) for data in self.datasets.values())