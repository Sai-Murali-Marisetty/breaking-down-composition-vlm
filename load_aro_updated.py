"""
Load ARO dataset from your downloaded files
Updated to match your actual file structure
"""

import os
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List
from PIL import Image

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

    def size(self, subset_name: str) -> int:
        return len(self.datasets.get(subset_name, []))

    def _resolve_image(self, ex):
        """Return a PIL.Image for common ARO fields."""
        if isinstance(ex, dict):
            if "image" in ex and isinstance(ex["image"], Image.Image):
                return ex["image"]
            if "image_path" in ex:
                p = Path(self.images_dir) / ex["image_path"]
                if p.exists():
                    return Image.open(p).convert("RGB")
        return None

    def _captions(self, ex):
        """Return (true_caption, false_caption) for common ARO schemas."""
        if not isinstance(ex, dict):
            return None, None
        tc = ex.get("true_caption") or ex.get("caption") or ex.get("statement_true")
        fc = ex.get("false_caption") or ex.get("negative_caption") or ex.get("statement_false")
        return tc, fc

    def get_example(self, idx: int, subset: str = None):
        """
        Backward-compatible accessor.
        If subset is given, index within that subset.
        If not, alternate across available subsets in order:
            vg_relation, vg_attribution.
        Returns: (subset_name, example_dict)
        """
        names = [n for n in ["vg_relation", "vg_attribution"] if n in self.datasets]
        if not names:
            raise IndexError("No ARO subsets loaded.")
        if subset:
            data = self.datasets.get(subset, [])
            if not data:
                raise IndexError(f"Subset {subset} empty/missing.")
            return subset, data[idx % len(data)]
        # round-robin across subsets
        s = names[idx % len(names)]
        pos = idx // len(names)
        data = self.datasets[s]
        return s, data[pos % len(data)]

    def iter_balanced(self, max_examples: int):
        """
        Yield (subset_name, example_dict) alternating between
        vg_relation and vg_attribution, capped at half each,
        and clipped by each subset’s available size.
        """
        rel = self.datasets.get("vg_relation", [])
        att = self.datasets.get("vg_attribution", [])
        n_each = max_examples // 2
        n_rel = min(n_each, len(rel))
        n_att = min(n_each, len(att))
        for i in range(max(n_rel, n_att)):
            if i < n_rel:
                yield "vg_relation", rel[i]
            if i < n_att:
                yield "vg_attribution", att[i]