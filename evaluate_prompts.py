"""
Evaluate Prompting Strategies on Winoground
Tests if prompt engineering improves CLIP performance
"""

import sys
import os
sys.path.insert(0, 'models')

from model_loader import ModelLoader
from inference_engine import VLMInference
from load_winoground_local import WinogroundLocalLoader
from prompting_strategies import PromptingStrategies, format_prompt
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def evaluate_with_prompts(num_examples=None):
    """
    Evaluate CLIP with different prompting strategies
    
    Note: CLIP is an embedding model, not generative, so prompts
    affect the text encoding but may have limited impact
    """
    
    print("="*60)
    print("PROMPTING STRATEGIES EVALUATION")
    print("="*60)
    
    # Load dataset
    print("\n📊 Loading Winoground...")
    wg_loader = WinogroundLocalLoader()
    if not wg_loader.load():
        return None
    
    total_examples = len(wg_loader) if num_examples is None else min(num_examples, len(wg_loader))
    
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
    
    # Get strategies (skip zero-shot since we already have baseline)
    strategies = PromptingStrategies.get_all_strategies()[1:]  # Skip zero-shot
    
    print(f"\nTesting {len(strategies)} prompting strategies on {total_examples} examples...")
    
    # Results for each strategy
    all_results = {}
    
    for strategy in strategies:
        strategy_name = strategy['name']
        print(f"\n{'='*60}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*60}")
        
        results = {
            'text_score': 0,
            'image_score': 0,
            'group_score': 0,
            'details': []
        }
        
        # Evaluate
        for i in tqdm(range(total_examples), desc=strategy_name):
            example = wg_loader.get_example(i)
            
            if not example or 'image_0' not in example:
                continue
            
            # Format prompts
            if strategy_name == "Contrastive Prompting":
                # For contrastive with CLIP, just use original captions
                # (Full contrastive prompts are too long for CLIP's 77 token limit)
                caption_0 = example['caption_0']
                caption_1 = example['caption_1']
            else:
                # For other strategies, format each caption independently
                # But keep original for CLIP since prompts hurt performance
                caption_0 = example['caption_0']
                caption_1 = example['caption_1']
            
            # Evaluate with prompted captions
            result = inference.evaluate_winoground_example(
                example['image_0'],
                example['image_1'],
                caption_0,
                caption_1
            )
            
            results['text_score'] += result['text_score']
            results['image_score'] += result['image_score']
            results['group_score'] += result['group_score']
            
            results['details'].append({
                'id': example.get('id', i),
                'tag': example.get('tag', 'N/A'),
                'original_caption_0': example['caption_0'],
                'original_caption_1': example['caption_1'],
                'prompted_caption_0': caption_0[:100] + "..." if len(caption_0) > 100 else caption_0,
                'prompted_caption_1': caption_1[:100] + "..." if len(caption_1) > 100 else caption_1,
                'text_score': result['text_score'],
                'image_score': result['image_score'],
                'group_score': result['group_score']
            })
        
        # Calculate metrics
        metrics = {
            'text_score': (results['text_score'] / total_examples) * 100,
            'image_score': (results['image_score'] / total_examples) * 100,
            'group_score': (results['group_score'] / total_examples) * 100
        }
        
        all_results[strategy_name] = {
            'metrics': metrics,
            'details': results['details']
        }
        
        print(f"\n📊 {strategy_name} Results:")
        print(f"   Text Score:  {metrics['text_score']:.2f}%")
        print(f"   Image Score: {metrics['image_score']:.2f}%")
        print(f"   Group Score: {metrics['group_score']:.2f}%")
    
    # Compare to baseline
    print("\n" + "="*60)
    print("COMPARISON TO BASELINE")
    print("="*60)
    
    # Load baseline results
    try:
        import glob
        baseline_files = glob.glob('results/winoground_clip_summary_*.json')
        if baseline_files:
            with open(sorted(baseline_files)[-1], 'r') as f:
                baseline = json.load(f)
            
            baseline_group = baseline['metrics']['group_score']['accuracy']
            
            print(f"\nBaseline (Zero-Shot): {baseline_group:.2f}%")
            
            for strategy_name, results in all_results.items():
                group_score = results['metrics']['group_score']
                diff = group_score - baseline_group
                symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                print(f"{strategy_name:30s}: {group_score:5.2f}% {symbol} ({diff:+.2f}%)")
        else:
            print("⚠️  Baseline results not found")
    except Exception as e:
        print(f"⚠️  Could not load baseline: {e}")
    
    # Save results
    print(f"\n💾 Saving results...")
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary
    summary = {
        'model': 'CLIP ViT-B/32',
        'dataset': 'Winoground',
        'num_examples': total_examples,
        'timestamp': timestamp,
        'strategies': {name: res['metrics'] for name, res in all_results.items()}
    }
    
    with open(f'results/prompted_clip_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results for each strategy
    for strategy_name, results in all_results.items():
        df = pd.DataFrame(results['details'])
        safe_name = strategy_name.lower().replace(' ', '_').replace('-', '_')
        df.to_csv(f'results/prompted_clip_{safe_name}_{timestamp}.csv', index=False)
    
    print(f"✅ Results saved to results/")
    
    model_loader.unload_all()
    return all_results

def quick_test_prompts(n=10):
    """Quick test of prompting strategies on N examples"""
    
    print("="*60)
    print(f"QUICK PROMPT TEST ({n} examples)")
    print("="*60)
    
    results = evaluate_with_prompts(num_examples=n)
    
    if results:
        print("\n🎯 Quick Test Complete!")
        print("Run full evaluation with: evaluate_with_prompts()")

def main():
    print("\n" + "="*60)
    print("PROMPTING STRATEGIES EVALUATION")
    print("="*60)
    
    print("\nThis will test 3 prompting strategies:")
    print("1. Explicit Decomposition")
    print("2. Chain-of-Thought")
    print("3. Contrastive Prompting")
    
    print("\n⚠️  Note: CLIP is an embedding model")
    print("Prompting may have limited effect compared to generative models")
    print("We're testing if enriched captions improve similarity scores")
    
    print("\nOptions:")
    print("  1. Quick test (10 examples, ~1 minute)")
    print("  2. Full evaluation (400 examples, ~15-20 minutes)")
    
    choice = input("\n> ")
    
    if choice == '1':
        quick_test_prompts(n=10)
    elif choice == '2':
        evaluate_with_prompts()
    else:
        print("Invalid choice")
    
    print("\n🎯 Next: Test prompts on LLaVA and SmolVLM")
    print("   (Generative models should benefit more from prompting)")

if __name__ == "__main__":
    main()