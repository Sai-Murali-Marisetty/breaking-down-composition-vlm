"""
Baseline Evaluation: CLIP on ARO
Evaluates CLIP ViT-B/32 on ARO VG-Relation and VG-Attribution
Saves detailed results for analysis
"""

import sys
import os

# Add parent directory to path (for load_aro_updated.py)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add models directory to path
models_dir = os.path.join(parent_dir, 'models')
sys.path.insert(0, models_dir)

from model_loader import ModelLoader
from inference_engine import VLMInference
from load_aro_updated import AROLoader
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def evaluate_clip_aro(save_results=True):
    """
    Run full CLIP evaluation on ARO dataset
    """
    
    print("="*60)
    print("BASELINE EVALUATION: CLIP ON ARO")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    print("\n📊 Loading ARO dataset...")
    aro_loader = AROLoader()
    if not aro_loader.load():
        print("❌ Failed to load dataset")
        return None
    
    # Load model
    print("\n🤖 Loading CLIP model...")
    model_loader = ModelLoader()
    if not model_loader.load_clip():
        print("❌ Failed to load CLIP")
        return None
    
    # Create inference engine
    inference = VLMInference(
        model=model_loader.get_model('clip'),
        processor=model_loader.get_processor('clip'),
        model_type='clip',
        device=model_loader.device
    )
    
    # Evaluate each subset
    all_results = {}
    
    for subset_name in ['vg_relation', 'vg_attribution']:
        subset_data = aro_loader.get_subset(subset_name)
        
        if not subset_data:
            print(f"\n⚠️  Skipping {subset_name} (not loaded)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {subset_name}")
        print(f"{'='*60}")
        print(f"Examples: {len(subset_data)}")
        
        results = {
            'correct': 0,
            'total': 0,
            'details': []
        }
        
        # Evaluate with progress bar
        for i, example in enumerate(tqdm(subset_data[:1000], desc=f"{subset_name}")):  # Limit to 1000 for speed
            # ARO format varies - adapt as needed
            try:
                # Try to extract image and captions
                # Format may vary - check your actual data structure
                if 'image' in example:
                    image = example['image']
                elif 'image_path' in example:
                    # Load image from path
                    img_path = os.path.join('data/aro/images', example['image_path'])
                    if os.path.exists(img_path):
                        image = Image.open(img_path)
                    else:
                        continue
                else:
                    continue
                
                # Get captions
                true_caption = example.get('true_caption', example.get('caption', ''))
                false_caption = example.get('false_caption', example.get('negative_caption', ''))
                
                if not true_caption or not false_caption:
                    continue
                
                # Evaluate
                result = inference.evaluate_aro_example(image, true_caption, false_caption)
                
                results['correct'] += result['correct']
                results['total'] += 1
                
                # Store details
                results['details'].append({
                    'id': i,
                    'true_caption': true_caption,
                    'false_caption': false_caption,
                    'correct': result['correct'],
                    'true_score': result['true_score'],
                    'false_score': result['false_score']
                })
                
            except Exception as e:
                # Skip problematic examples
                continue
        
        # Calculate accuracy
        if results['total'] > 0:
            accuracy = (results['correct'] / results['total']) * 100
            
            print(f"\n📊 {subset_name} Results:")
            print(f"   Correct: {results['correct']}/{results['total']}")
            print(f"   Accuracy: {accuracy:.2f}%")
            
            all_results[subset_name] = {
                'correct': results['correct'],
                'total': results['total'],
                'accuracy': accuracy,
                'details': results['details']
            }
        else:
            print(f"\n⚠️  No valid examples evaluated for {subset_name}")
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    
    total_correct = sum(r['correct'] for r in all_results.values())
    total_examples = sum(r['total'] for r in all_results.values())
    overall_accuracy = (total_correct / total_examples * 100) if total_examples > 0 else 0
    
    print(f"Total: {total_correct}/{total_examples} ({overall_accuracy:.2f}%)")
    
    # Save results
    if save_results:
        print(f"\n💾 Saving results...")
        os.makedirs('results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary = {
            'model': 'CLIP ViT-B/32',
            'dataset': 'ARO',
            'timestamp': timestamp,
            'subsets': {k: {
                'correct': v['correct'],
                'total': v['total'],
                'accuracy': v['accuracy']
            } for k, v in all_results.items()},
            'overall': {
                'correct': total_correct,
                'total': total_examples,
                'accuracy': overall_accuracy
            }
        }
        
        summary_file = f'results/aro_clip_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary: {summary_file}")
        
        # Save detailed results for each subset
        for subset_name, subset_results in all_results.items():
            df = pd.DataFrame(subset_results['details'])
            csv_file = f'results/aro_clip_{subset_name}_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            print(f"✅ {subset_name}: {csv_file}")
    
    # Clean up
    model_loader.unload_all()
    
    print(f"\n✅ Evaluation complete!")
    return all_results

def main():
    print("\n" + "="*60)
    print("ARO BASELINE EVALUATION - CLIP")
    print("="*60)
    
    print("\nThis will:")
    print("1. Load ARO VG-Relation and VG-Attribution")
    print("2. Evaluate CLIP on subset of examples")
    print("3. Calculate accuracy per subset")
    print("4. Save detailed results")
    
    print("\n⏱️  Estimated time: ~10-15 minutes")
    print("💾 Will save results to: results/")
    
    input("\nPress Enter to start baseline evaluation...")
    
    results = evaluate_clip_aro(save_results=True)
    
    if results:
        print("\n🎉 BASELINE EVALUATION COMPLETE!")
    else:
        print("\n❌ Evaluation failed")

if __name__ == "__main__":
    main()