"""
Baseline Evaluation: CLIP on Winoground
Evaluates CLIP ViT-B/32 on all 400 Winoground examples
Saves detailed results for analysis
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add models directory to path  
models_dir = os.path.join(parent_dir, 'models')
sys.path.insert(0, models_dir)

from model_loader import ModelLoader
from inference_engine import VLMInference
from load_winoground_local import WinogroundLocalLoader
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def evaluate_clip_winoground(save_results=True):
    """
    Run full CLIP evaluation on Winoground
    """
    
    print("="*60)
    print("BASELINE EVALUATION: CLIP ON WINOGROUND")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    print("\n📊 Loading Winoground dataset...")
    wg_loader = WinogroundLocalLoader()
    if not wg_loader.load():
        print("❌ Failed to load dataset")
        return None
    
    total_examples = len(wg_loader)
    print(f"✅ Loaded {total_examples} examples")
    
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
    
    # Evaluation loop
    print(f"\n🔄 Evaluating {total_examples} examples...")
    print("="*60)
    
    results = {
        'text_score': 0,
        'image_score': 0,
        'group_score': 0,
        'details': []
    }
    
    # Progress bar
    for i in tqdm(range(total_examples), desc="Evaluating"):
        example = wg_loader.get_example(i)
        
        if not example or 'image_0' not in example or 'image_1' not in example:
            print(f"\n⚠️  Example {i}: Images not loaded, skipping")
            continue
        
        # Evaluate
        result = inference.evaluate_winoground_example(
            example['image_0'],
            example['image_1'],
            example['caption_0'],
            example['caption_1']
        )
        
        # Update aggregate scores
        results['text_score'] += result['text_score']
        results['image_score'] += result['image_score']
        results['group_score'] += result['group_score']
        
        # Store detailed results
        results['details'].append({
            'id': example.get('id', i),
            'caption_0': example['caption_0'],
            'caption_1': example['caption_1'],
            'tag': example.get('tag', 'N/A'),
            'text_score': result['text_score'],
            'image_score': result['image_score'],
            'group_score': result['group_score'],
            'scores_img0_cap0': float(result['scores_img0'][0]),
            'scores_img0_cap1': float(result['scores_img0'][1]),
            'scores_img1_cap0': float(result['scores_img1'][0]),
            'scores_img1_cap1': float(result['scores_img1'][1]),
        })
    
    # Calculate final metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    metrics = {}
    for metric in ['text_score', 'image_score', 'group_score']:
        score = results[metric]
        percentage = (score / total_examples) * 100
        metrics[metric] = {
            'correct': int(score),
            'total': total_examples,
            'accuracy': percentage
        }
        print(f"{metric:15s}: {score:3.0f}/{total_examples} ({percentage:5.2f}%)")
    
    # Save results
    if save_results:
        print(f"\n💾 Saving results...")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary = {
            'model': 'CLIP ViT-B/32',
            'model_id': 'openai/clip-vit-base-patch32',
            'dataset': 'Winoground',
            'num_examples': total_examples,
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        summary_file = f'results/winoground_clip_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary: {summary_file}")
        
        # Save detailed results as CSV
        df = pd.DataFrame(results['details'])
        csv_file = f'results/winoground_clip_detailed_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"✅ Detailed results: {csv_file}")
        
        # Save by tag analysis
        tag_analysis = df.groupby('tag').agg({
            'text_score': ['sum', 'count', 'mean'],
            'image_score': ['sum', 'count', 'mean'],
            'group_score': ['sum', 'count', 'mean']
        }).round(3)
        
        tag_file = f'results/winoground_clip_by_tag_{timestamp}.csv'
        tag_analysis.to_csv(tag_file)
        print(f"✅ By-tag analysis: {tag_file}")
    
    # Compare to baselines
    print("\n" + "="*60)
    print("COMPARISON TO BASELINES")
    print("="*60)
    print(f"Random Chance:        25.00%")
    print(f"CLIP (Published):    ~30.00%")
    print(f"Your CLIP:           {metrics['group_score']['accuracy']:5.2f}%")
    print(f"Human Performance:   >95.00%")
    
    # Clean up
    model_loader.unload_all()
    
    print(f"\n✅ Evaluation complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return metrics

def main():
    print("\n" + "="*60)
    print("WINOGROUND BASELINE EVALUATION - CLIP")
    print("="*60)
    
    print("\nThis will:")
    print("1. Load all 400 Winoground examples")
    print("2. Evaluate CLIP on each example")
    print("3. Calculate text, image, and group scores")
    print("4. Save detailed results to results/")
    print("5. Provide analysis by compositional tag")
    
    print("\n⏱️  Estimated time: ~5-10 minutes")
    print("💾 Will save results to: results/")
    
    input("\nPress Enter to start baseline evaluation...")
    
    metrics = evaluate_clip_winoground(save_results=True)
    
    if metrics:
        print("\n" + "="*60)
        print("🎉 BASELINE EVALUATION COMPLETE!")
        print("="*60)
        
        print("\n📊 Results Summary:")
        print(f"   Group Score: {metrics['group_score']['accuracy']:.2f}%")
        print(f"   Text Score:  {metrics['text_score']['accuracy']:.2f}%")
        print(f"   Image Score: {metrics['image_score']['accuracy']:.2f}%")
        
        print("\n📁 Results saved to results/")
        print("   - Summary JSON file")
        print("   - Detailed CSV file")
        print("   - By-tag analysis CSV")
        
        print("\n🎯 Next Steps:")
        print("1. Analyze results by compositional tag")
        print("2. Test LLaVA and SmolVLM models")
        print("3. Implement prompting strategies")
        print("4. Compare baseline vs prompted performance")
    else:
        print("\n❌ Evaluation failed - check error messages")

if __name__ == "__main__":
    main()