"""
Baseline Evaluation: LLaVA on Winoground
Tests LLaVA-1.5-7B on compositional reasoning
"""

import sys
import os
sys.path.insert(0, 'models')

from model_loader import ModelLoader
from load_winoground_local import WinogroundLocalLoader
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import torch

def evaluate_llava_example(model, processor, image_0, image_1, caption_0, caption_1, device):
    """
    Evaluate a single Winoground example with LLaVA
    
    LLaVA is generative, so we'll ask it to:
    1. Describe each image
    2. Match descriptions to captions
    """
    
    results = {
        'text_score': 0,
        'image_score': 0,
        'group_score': 0,
        'responses': {}
    }
    
    # Prepare conversation template
    def generate_response(image, prompt):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0
            )
        
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response
    
    # Test 1: Ask which caption matches image 0
    prompt_img0 = f"Which caption better describes this image?\nA) {caption_0}\nB) {caption_1}\nAnswer with just 'A' or 'B'."
    response_img0 = generate_response(image_0, prompt_img0)
    
    # Test 2: Ask which caption matches image 1  
    prompt_img1 = f"Which caption better describes this image?\nA) {caption_0}\nB) {caption_1}\nAnswer with just 'A' or 'B'."
    response_img1 = generate_response(image_1, prompt_img1)
    
    # Parse responses
    # Image 0 should prefer Caption 0 (answer A)
    # Image 1 should prefer Caption 1 (answer B)
    
    img0_correct = 'A' in response_img0.upper()[:10]  # Check first 10 chars
    img1_correct = 'B' in response_img1.upper()[:10]
    
    # Calculate scores
    if img0_correct and img1_correct:
        results['text_score'] = 1
        results['image_score'] = 1
        results['group_score'] = 1
    elif img0_correct or img1_correct:
        # Partial credit
        if img0_correct:
            results['image_score'] = 0.5
        if img1_correct:
            results['image_score'] = 0.5
    
    results['responses'] = {
        'image_0_response': response_img0,
        'image_1_response': response_img1,
        'image_0_correct': img0_correct,
        'image_1_correct': img1_correct
    }
    
    return results

def evaluate_llava_winoground(num_examples=None):
    """
    Run LLaVA evaluation on Winoground
    """
    
    print("="*60)
    print("BASELINE EVALUATION: LLaVA ON WINOGROUND")
    print("="*60)
    print("⚠️  This is a generative model - evaluation is slower!")
    
    # Load dataset
    print("\n📊 Loading Winoground...")
    wg_loader = WinogroundLocalLoader()
    if not wg_loader.load():
        return None
    
    total_examples = len(wg_loader) if num_examples is None else min(num_examples, len(wg_loader))
    
    # Load model
    print("\n🤖 Loading LLaVA-1.5-7B...")
    print("⚠️  This model is ~14GB and will take a few minutes to load...")
    
    model_loader = ModelLoader()
    if not model_loader.load_llava():
        return None
    
    model = model_loader.get_model('llava')
    processor = model_loader.get_processor('llava')
    device = model_loader.device
    
    # Evaluation
    print(f"\n🔄 Evaluating {total_examples} examples...")
    print("⏱️  This will take ~1-2 minutes per example")
    print(f"   Estimated time: {total_examples * 1.5 / 60:.1f} - {total_examples * 2 / 60:.1f} minutes")
    
    results = {
        'text_score': 0,
        'image_score': 0,
        'group_score': 0,
        'details': []
    }
    
    for i in tqdm(range(total_examples), desc="Evaluating"):
        example = wg_loader.get_example(i)
        
        if not example or 'image_0' not in example:
            continue
        
        try:
            result = evaluate_llava_example(
                model, processor,
                example['image_0'],
                example['image_1'],
                example['caption_0'],
                example['caption_1'],
                device
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
                'response_img0': result['responses']['image_0_response'],
                'response_img1': result['responses']['image_1_response']
            })
            
        except Exception as e:
            print(f"\n⚠️  Error on example {i}: {e}")
            continue
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    metrics = {}
    for metric in ['text_score', 'image_score', 'group_score']:
        score = results[metric]
        pct = (score / total_examples) * 100
        metrics[metric] = pct
        print(f"{metric:15s}: {score:5.1f}/{total_examples} ({pct:5.2f}%)")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    summary = {
        'model': 'LLaVA-1.5-7B',
        'dataset': 'Winoground',
        'num_examples': total_examples,
        'timestamp': timestamp,
        'metrics': metrics
    }
    
    with open(f'results/winoground_llava_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    df = pd.DataFrame(results['details'])
    df.to_csv(f'results/winoground_llava_{timestamp}.csv', index=False)
    
    print(f"\n💾 Results saved to results/")
    
    # Compare to CLIP
    print("\n" + "="*60)
    print("COMPARISON TO CLIP")
    print("="*60)
    print(f"CLIP Group Score:  31.25%")
    print(f"LLaVA Group Score: {metrics['group_score']:.2f}%")
    
    diff = metrics['group_score'] - 31.25
    if diff > 0:
        print(f"📈 LLaVA is {diff:.2f}% better!")
    else:
        print(f"📉 LLaVA is {abs(diff):.2f}% worse")
    
    model_loader.unload_all()
    return metrics

def main():
    print("\n" + "="*60)
    print("LLaVA BASELINE EVALUATION")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Quick test (10 examples, ~15-20 min)")
    print("  2. Subset (50 examples, ~1.5-2 hours)")
    print("  3. Full evaluation (400 examples, ~10-13 hours)")
    
    choice = input("\n> ")
    
    if choice == '1':
        print("\n🚀 Running quick test...")
        evaluate_llava_winoground(num_examples=10)
    elif choice == '2':
        print("\n🚀 Running subset evaluation...")
        evaluate_llava_winoground(num_examples=50)
    elif choice == '3':
        confirm = input("\n⚠️  Full evaluation will take 10-13 hours. Continue? (y/n): ")
        if confirm.lower() == 'y':
            evaluate_llava_winoground()
        else:
            print("Cancelled.")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()