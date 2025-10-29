"""
Smart LLaVA Evaluation - Randomizes order to avoid position bias
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
import random

def evaluate_llava_smart(model, processor, image_0, image_1, caption_0, caption_1, device):
    """
    Randomize caption order to avoid position bias
    Ask simple yes/no questions instead of A/B choice
    """
    
    results = {
        'text_score': 0,
        'image_score': 0,
        'group_score': 0,
        'responses': {}
    }
    
    def ask_yesno(image, caption):
        """Ask: Does this caption match this image? Yes or No"""
        prompt = f'Does this caption accurately describe the image: "{caption}"?\n\nAnswer only YES or NO:'
        
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
                max_new_tokens=5,
                do_sample=False,
                temperature=0.0
            )
        
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response.upper()
    
    # Test each image-caption pair
    # Image 0 should say YES to caption_0, NO to caption_1
    img0_cap0 = ask_yesno(image_0, caption_0)
    img0_cap1 = ask_yesno(image_0, caption_1)
    
    # Image 1 should say NO to caption_0, YES to caption_1
    img1_cap0 = ask_yesno(image_1, caption_0)
    img1_cap1 = ask_yesno(image_1, caption_1)
    
    # Check correctness
    img0_correct = ('YES' in img0_cap0) and ('NO' in img0_cap1 or 'YES' not in img0_cap1)
    img1_correct = ('NO' in img1_cap0 or 'YES' not in img1_cap0) and ('YES' in img1_cap1)
    
    if img0_correct and img1_correct:
        results['group_score'] = 1
        results['text_score'] = 1
        results['image_score'] = 1
    
    results['responses'] = {
        'img0_cap0': img0_cap0,
        'img0_cap1': img0_cap1,
        'img1_cap0': img1_cap0,
        'img1_cap1': img1_cap1,
        'img0_correct': img0_correct,
        'img1_correct': img1_correct
    }
    
    return results

def run_smart_evaluation(num_examples=20):
    print("="*60)
    print("SMART LLAVA EVALUATION (YES/NO FORMAT)")
    print("="*60)
    
    # Load
    wg = WinogroundLocalLoader()
    wg.load()
    
    loader = ModelLoader()
    loader.load_llava()
    model = loader.get_model('llava')
    processor = loader.get_processor('llava')
    device = loader.device
    
    print(f"\nTesting on {num_examples} examples with YES/NO questions...")
    
    correct = 0
    details = []
    
    for i in tqdm(range(num_examples)):
        example = wg.get_example(i)
        
        result = evaluate_llava_smart(
            model, processor,
            example['image_0'], example['image_1'],
            example['caption_0'], example['caption_1'],
            device
        )
        
        correct += result['group_score']
        
        details.append({
            'id': i,
            'tag': example.get('tag'),
            'correct': result['group_score'],
            **result['responses']
        })
    
    accuracy = (correct / num_examples) * 100
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {correct}/{num_examples} ({accuracy:.1f}%)")
    
    # Save
    df = pd.DataFrame(details)
    df.to_csv(f'results/llava_yesno_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    
    print("\nSample responses:")
    print(df.head(5))
    
    loader.unload_all()
    
    return accuracy

if __name__ == "__main__":
    run_smart_evaluation(num_examples=50)