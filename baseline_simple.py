"""
Simplified Baseline Evaluation - Run from Root Directory
"""

import sys
import os

# Ensure we're in the right directory
sys.path.insert(0, 'models')
sys.path.insert(0, '.')

from model_loader import ModelLoader
from inference_engine import VLMInference
from load_winoground_local import WinogroundLocalLoader
from load_aro_updated import AROLoader
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def evaluate_winoground():
    """Evaluate CLIP on Winoground"""
    
    print("="*60)
    print("WINOGROUND EVALUATION")
    print("="*60)
    
    # Load dataset
    print("\n📊 Loading Winoground...")
    wg_loader = WinogroundLocalLoader()
    if not wg_loader.load():
        return None
    
    total_examples = len(wg_loader)
    
    # Load model
    print("\n🤖 Loading CLIP...")
    model_loader = ModelLoader()
    if not model_loader.load_clip():
        return None
    
    inference = VLMInference(
        model=model_loader.get_model('clip'),
        processor=model_loader.get_processor('clip'),
        model_type='clip',
        device=model_loader.device
    )
    
    # Evaluate
    print(f"\n🔄 Evaluating {total_examples} examples...")
    
    results = {
        'text_score': 0,
        'image_score': 0,
        'group_score': 0,
        'details': []
    }
    
    for i in tqdm(range(total_examples), desc="Progress"):
        example = wg_loader.get_example(i)
        
        if not example or 'image_0' not in example:
            continue
        
        result = inference.evaluate_winoground_example(
            example['image_0'],
            example['image_1'],
            example['caption_0'],
            example['caption_1']
        )
        
        results['text_score'] += result['text_score']
        results['image_score'] += result['image_score']
        results['group_score'] += result['group_score']
        
        results['details'].append({
            'id': example.get('id', i),
            'caption_0': example['caption_0'],
            'caption_1': example['caption_1'],
            'tag': example.get('tag', 'N/A'),
            'text_score': result['text_score'],
            'image_score': result['image_score'],
            'group_score': result['group_score'],
        })
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for metric in ['text_score', 'image_score', 'group_score']:
        score = results[metric]
        pct = (score / total_examples) * 100
        print(f"{metric:15s}: {score:3.0f}/{total_examples} ({pct:5.2f}%)")
    
    # Save
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    summary = {
        'model': 'CLIP ViT-B/32',
        'dataset': 'Winoground',
        'num_examples': total_examples,
        'timestamp': timestamp,
        'text_score': float(results['text_score']),
        'image_score': float(results['image_score']),
        'group_score': float(results['group_score']),
        'text_accuracy': float(results['text_score'] / total_examples * 100),
        'image_accuracy': float(results['image_score'] / total_examples * 100),
        'group_accuracy': float(results['group_score'] / total_examples * 100),
    }
    
    with open(f'results/winoground_clip_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    df = pd.DataFrame(results['details'])
    df.to_csv(f'results/winoground_clip_{timestamp}.csv', index=False)
    
    print(f"\n💾 Results saved to results/")
    
    model_loader.unload_all()
    return summary

def evaluate_aro(max_per_subset=1000):
    """Evaluate CLIP on ARO (limited examples)"""
    
    print("\n" + "="*60)
    print("ARO EVALUATION")
    print("="*60)
    
    # Load dataset
    print("\n📊 Loading ARO...")
    aro_loader = AROLoader()
    if not aro_loader.load():
        return None
    
    # Load model
    print("\n🤖 Loading CLIP...")
    model_loader = ModelLoader()
    if not model_loader.load_clip():
        return None
    
    inference = VLMInference(
        model=model_loader.get_model('clip'),
        processor=model_loader.get_processor('clip'),
        model_type='clip',
        device=model_loader.device
    )
    
    all_results = {}
    
    for subset_name in ['vg_relation', 'vg_attribution']:
        subset_data = aro_loader.get_subset(subset_name)
        
        if not subset_data:
            continue
        
        print(f"\n🔄 Evaluating {subset_name} ({min(len(subset_data), max_per_subset)} examples)...")
        
        correct = 0
        total = 0
        
        for example in tqdm(subset_data[:max_per_subset], desc=subset_name):
            try:
                # ARO format - adapt to your actual data structure
                # This is a simplified version - you may need to adjust
                
                # Skip if missing required fields
                if 'image' not in example:
                    continue
                
                image = example['image']
                true_caption = example.get('caption', '')
                false_caption = example.get('negative_caption', '')
                
                if not true_caption or not false_caption:
                    continue
                
                result = inference.evaluate_aro_example(image, true_caption, false_caption)
                
                correct += result['correct']
                total += 1
                
            except:
                continue
        
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"\n{subset_name}: {correct}/{total} ({accuracy:.2f}%)")
            
            all_results[subset_name] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
    
    # Save
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'results/aro_clip_{timestamp}.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Results saved to results/")
    
    model_loader.unload_all()
    return all_results

def main():
    print("="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    
    print("\nChoose evaluation:")
    print("  1. Winoground only (~5-10 min)")
    print("  2. ARO only (~10-15 min)")
    print("  3. Both (~20-25 min)")
    
    choice = input("\n> ")
    
    if choice in ['1', '3']:
        results = evaluate_winoground()
        if results:
            print(f"\n✅ Winoground: {results['group_accuracy']:.2f}% group score")
    
    if choice in ['2', '3']:
        if choice == '3':
            input("\nPress Enter for ARO evaluation...")
        results = evaluate_aro()
        if results:
            print(f"\n✅ ARO evaluation complete")
    
    print("\n🎉 Evaluation complete! Check results/ directory")

if __name__ == "__main__":
    main()