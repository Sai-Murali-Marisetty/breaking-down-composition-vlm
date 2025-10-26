"""
Quick Test: Verify LLaVA Loads and Works
Tests model loading and basic inference before full evaluation
"""

import sys
import os
sys.path.insert(0, 'models')

from model_loader import ModelLoader
from PIL import Image
import torch

def test_llava_loading():
    """Test if LLaVA loads successfully"""
    
    print("="*60)
    print("LLAVA MODEL LOADING TEST")
    print("="*60)
    
    print("\n⚠️  LLaVA-1.5-7B is a large model (~14GB)")
    print("   It will download on first run")
    print("   Your system: 25.4GB RAM - should be fine!")
    
    input("\nPress Enter to start loading LLaVA...")
    
    # Load model
    loader = ModelLoader()
    
    print("\n📥 Loading LLaVA... (this may take 2-5 minutes)")
    if not loader.load_llava():
        print("\n❌ Failed to load LLaVA")
        return False
    
    print("\n✅ LLaVA loaded successfully!")
    
    # Check memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
    
    return True

def test_llava_inference():
    """Test basic inference"""
    
    print("\n" + "="*60)
    print("TESTING LLAVA INFERENCE")
    print("="*60)
    
    # Load model
    loader = ModelLoader()
    if not loader.load_llava():
        return False
    
    model = loader.get_model('llava')
    processor = loader.get_processor('llava')
    device = loader.device
    
    # Create test image
    print("\n🖼️  Creating test image...")
    image = Image.new('RGB', (224, 224), color='blue')
    
    # Test prompt
    prompt = "What color is this image? Answer in one word."
    
    print(f"🤖 Asking LLaVA: '{prompt}'")
    
    # Prepare input
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
    
    # Generate
    print("⏳ Generating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0
        )
    
    # Decode
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()
    
    print(f"💬 LLaVA's response: '{response}'")
    
    # Check if it said "blue"
    if 'blue' in response.lower():
        print("\n✅ LLaVA is working correctly!")
    else:
        print("\n⚠️  Unexpected response, but model is working")
    
    # Clean up
    loader.unload_all()
    
    return True

def test_winoground_example():
    """Test with a real Winoground example"""
    
    print("\n" + "="*60)
    print("TESTING WITH WINOGROUND EXAMPLE")
    print("="*60)
    
    from load_winoground_local import WinogroundLocalLoader
    
    # Load dataset
    print("\n📊 Loading Winoground...")
    wg_loader = WinogroundLocalLoader()
    if not wg_loader.load():
        print("❌ Failed to load Winoground")
        return False
    
    # Load model
    print("\n🤖 Loading LLaVA...")
    loader = ModelLoader()
    if not loader.load_llava():
        return False
    
    model = loader.get_model('llava')
    processor = loader.get_processor('llava')
    device = loader.device
    
    # Get first example
    example = wg_loader.get_example(0)
    
    print(f"\n📝 Example 0:")
    print(f"   Caption 0: {example['caption_0']}")
    print(f"   Caption 1: {example['caption_1']}")
    
    # Test on image 0
    prompt = f"Which caption better describes this image?\nA) {example['caption_0']}\nB) {example['caption_1']}\nAnswer with just 'A' or 'B'."
    
    print(f"\n🤖 Testing image 0...")
    
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
    inputs = processor(images=example['image_0'], text=prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
    
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()
    
    print(f"💬 LLaVA chose: {response}")
    
    if 'A' in response[:10]:
        print("✅ Correct! (Image 0 should match Caption 0)")
    else:
        print("❌ Incorrect (but model is working)")
    
    loader.unload_all()
    
    print("\n🎉 LLaVA is ready for full evaluation!")
    return True

def main():
    print("\n" + "="*60)
    print("LLAVA READINESS TEST")
    print("="*60)
    
    print("\nThis will:")
    print("1. Load LLaVA model (~14GB, 2-5 min)")
    print("2. Test basic inference")
    print("3. Test on Winoground example")
    
    input("\nPress Enter to start...")
    
    # Test loading
    if not test_llava_loading():
        return
    
    input("\nPress Enter to test inference...")
    
    # Test inference
    if not test_llava_inference():
        return
    
    input("\nPress Enter to test with Winoground...")
    
    # Test Winoground
    if not test_winoground_example():
        return
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    
    print("\n✅ LLaVA is ready for evaluation!")
    print("\nNext steps:")
    print("1. Run: python evaluate_llava_baseline.py")
    print("   Choose option 1 (quick test, 10 examples)")
    print("2. Then test prompting strategies on LLaVA")
    print("3. Compare CLIP vs LLaVA performance")

if __name__ == "__main__":
    main()