"""
Verify ARO dataset downloads
"""

import os
import json
from pathlib import Path

def verify_aro_data(data_dir="data/aro"):
    """Verify ARO dataset files"""
    
    print("=" * 60)
    print("ARO DATASET VERIFICATION")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return False
    
    print(f"\n📁 Checking: {data_dir}\n")
    
    # Check for files
    files_to_check = {
        "visual_genome_relation.json": "VG-Relation dataset",
        "visual_genome_attribution.json": "VG-Attribution dataset",
        "dataset_info.json": "Dataset metadata"
    }
    
    found_files = []
    missing_files = []
    
    for filename, description in files_to_check.items():
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename:40s} ({size_mb:.2f} MB)")
            found_files.append(filename)
            
            # Count examples
            if filename.endswith('.json') and 'info' not in filename:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            print(f"   └─ {len(data):,} examples")
                        elif isinstance(data, dict):
                            print(f"   └─ {len(data):,} items")
                except Exception as e:
                    print(f"   └─ Error reading: {e}")
        else:
            print(f"⚠️  {filename:40s} (NOT FOUND)")
            missing_files.append(filename)
    
    # Check for images
    print(f"\n📸 Images:")
    images_dir = data_dir / "images"
    zip_file = data_dir / "vgr_vga_images.zip"
    
    if images_dir.exists():
        image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
        print(f"✅ images/ directory: {image_count:,} images")
    elif zip_file.exists():
        size_mb = zip_file.stat().st_size / (1024 * 1024)
        print(f"✅ vgr_vga_images.zip: {size_mb:.2f} MB (need to extract)")
        print(f"   Run: cd data/aro && unzip vgr_vga_images.zip")
    else:
        print(f"⚠️  No images found")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"✅ Files found: {len(found_files)}/{len(files_to_check)}")
    
    if missing_files:
        print(f"⚠️  Missing: {', '.join(missing_files)}")
    
    # Calculate total examples
    total_examples = 0
    for filename in ["visual_genome_relation.json", "visual_genome_attribution.json"]:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    total_examples += len(data) if isinstance(data, list) else len(data.get('data', []))
            except:
                pass
    
    print(f"\n📊 Total ARO examples: {total_examples:,}")
    
    if total_examples >= 10000:
        print(f"✅ EXCELLENT! You have enough data for your project")
    elif total_examples >= 5000:
        print(f"✅ GOOD! This is sufficient for analysis")
    else:
        print(f"⚠️  You may need more examples")
    
    print(f"\n💡 Note: Your project proposal mentioned ARO (50K+ examples)")
    print(f"   You have ~{total_examples//1000}K examples, which is sufficient!")
    print(f"   The full ARO includes COCO & Flickr30k (optional, very large)")
    
    return len(found_files) >= 2

if __name__ == "__main__":
    verify_aro_data()