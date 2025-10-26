"""
Test CLIP with real Winoground data
Evaluate on a few examples to verify everything works
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from model_loader import ModelLoader
from inference_engine import VLMInference
from load_winoground_local import WinogroundLocalLoader

def test_winoground_examples(num_examples=5):
    """Test CLIP on first N Winoground examples"""
    
    print("="*60)
    print("TESTING WITH REAL WINOGROUND DATA")
    print("="*60)
    
    # Load Winoground dataset
    print("\n1. Loading Winoground dataset...")
    wg_loader = WinogroundLocalLoader()
    if not wg_loader.load():
        print("❌ Failed to load Winoground data")
        print("Make sure data is in data/winoground/")
        return False
    
    # Load CLIP model
    print("\n2. Loading CLIP model...")
    model_loader = ModelLoader()
    if not model_loader.load_clip():
        print("❌ Failed to load CLIP")
        return False
    
    # Create inference engine
    inference = VLMInference(
        model=model_loader.get_model('clip'),
        processor=model_loader.get_processor('clip'),
        model_type='clip',
        device=model_loader.device
    )
    
    # Evaluate examples
    print(f"\n3. Evaluating first {num_examples} examples...")
    print("="*60)
    
    results = {
        'text_score': 0,
        'image_score': 0,
        'group_score': 0
    }
    
    for i in range(min(num_examples, len(wg_loader))):
        example = wg_loader.get_example(i)
        
        if not example or 'image_0' not in example:
            print(f"\nExample {i}: ⚠️  Images not loaded, skipping")
            continue
        
        print(f"\n--- Example {i} ---")
        print(f"Caption 0: {example['caption_0']}")
        print(f"Caption 1: {example['caption_1']}")
        
        # Evaluate
        result = inference.evaluate_winoground_example(
            example['image_0'],
            example['image_1'],
            example['caption_0'],
            example['caption_1']
        )
        
        # Update scores
        results['text_score'] += result['text_score']
        results['image_score'] += result['image_score']
        results['group_score'] += result['group_score']
        
        # Show results
        print(f"Text Score:  {'✅' if result['text_score'] else '❌'}")
        print(f"Image Score: {'✅' if result['image_score'] else '❌'}")
        print(f"Group Score: {'✅' if result['group_score'] else '❌'}")
        
        # Show detailed scores
        print(f"\nDetailed scores:")
        print(f"  Image 0 → Caption 0: {result['scores_img0'][0]:.3f} vs Caption 1: {result['scores_img0'][1]:.3f}")
        print(f"  Image 1 → Caption 0: {result['scores_img1'][0]:.3f} vs Caption 1: {result['scores_img1'][1]:.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for metric in ['text_score', 'image_score', 'group_score']:
        accuracy = (results[metric] / num_examples) * 100
        print(f"{metric:15s}: {results[metric]}/{num_examples} ({accuracy:.1f}%)")
    
    print("\n💡 Note: Winoground is VERY hard!")
    print("   - Random chance: 25% group score")
    print("   - CLIP typically gets ~30% group score")
    print("   - Humans get >95%")
    
    # Clean up
    model_loader.unload_all()
    
    return True

def main():
    print("\n" + "="*60)
    print("WINOGROUND EVALUATION TEST")
    print("="*60)
    
    print("\nThis will:")
    print("1. Load your Winoground dataset")
    print("2. Load CLIP model")
    print("3. Evaluate on 5 examples")
    print("4. Show detailed results")
    
    input("\nPress Enter to start...")
    
    success = test_winoground_examples(num_examples=5)
    
    if success:
        print("\n" + "="*60)
        print("🎉 TEST COMPLETE!")
        print("="*60)
        print("\n✅ Everything is working!")
        print("\nYou're ready to:")
        print("1. Run full baseline evaluation on all 400 examples")
        print("2. Test LLaVA and SmolVLM models")
        print("3. Implement prompting strategies")
    else:
        print("\n❌ Test failed - check error messages above")

if __name__ == "__main__":
    main()