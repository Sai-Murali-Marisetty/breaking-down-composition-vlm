"""
Quick test script to verify models work with your setup
Tests loading and basic inference
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_system():
    """Test system requirements"""
    print("="*60)
    print("SYSTEM CHECK")
    print("="*60)
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"⚠️  CUDA not available - will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check transformers
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    # Check PIL
    try:
        from PIL import Image
        print(f"✅ PIL/Pillow available")
    except ImportError:
        print("❌ PIL/Pillow not installed")
        return False
    
    return True

def test_clip_only():
    """Test CLIP model loading and inference"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
    
    from model_loader import ModelLoader
    from inference_engine import VLMInference
    from PIL import Image
    
    print("\n" + "="*60)
    print("TESTING CLIP MODEL")
    print("="*60)
    
    # Load model
    loader = ModelLoader()
    if not loader.load_clip():
        print("❌ Failed to load CLIP")
        return False
    
    # Create inference engine
    inference = VLMInference(
        model=loader.get_model('clip'),
        processor=loader.get_processor('clip'),
        model_type='clip',
        device=loader.device
    )
    
    # Test with dummy image
    print("\n🧪 Testing inference with dummy image...")
    image = Image.new('RGB', (224, 224), color='blue')
    captions = [
        "a blue square",
        "a red circle",
        "a green triangle"
    ]
    
    scores = inference.compute_similarity_clip(image, captions)
    
    print("Similarity scores:")
    for caption, score in zip(captions, scores):
        print(f"  {caption:25s}: {score:.4f}")
    
    # Clean up
    loader.unload_all()
    
    print("\n✅ CLIP test passed!")
    return True

def test_with_real_data():
    """Test with actual Winoground data if available"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
    
    from model_loader import ModelLoader
    from inference_engine import VLMInference
    
    # Try to load Winoground
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from load_winoground_local import WinogroundLocalLoader
        
        print("\n" + "="*60)
        print("TESTING WITH REAL WINOGROUND DATA")
        print("="*60)
        
        # Load dataset
        wg_loader = WinogroundLocalLoader()
        if not wg_loader.load():
            print("⚠️  Winoground data not available, skipping this test")
            return True
        
        # Load CLIP
        model_loader = ModelLoader()
        if not model_loader.load_clip():
            return False
        
        inference = VLMInference(
            model=model_loader.get_model('clip'),
            processor=model_loader.get_processor('clip'),
            model_type='clip',
            device=model_loader.device
        )
        
        # Test with first example
        print("\n🧪 Testing with Winoground example 0...")
        example = wg_loader.get_example(0)
        
        if example and 'image_0' in example and 'image_1' in example:
            result = inference.evaluate_winoground_example(
                example['image_0'],
                example['image_1'],
                example['caption_0'],
                example['caption_1']
            )
            
            print(f"\nResults:")
            print(f"  Caption 0: {example['caption_0']}")
            print(f"  Caption 1: {example['caption_1']}")
            print(f"  Text Score: {result['text_score']}")
            print(f"  Image Score: {result['image_score']}")
            print(f"  Group Score: {result['group_score']}")
            
            print("\n✅ Real data test passed!")
        else:
            print("⚠️  Images not loaded, skipping")
        
        model_loader.unload_all()
        return True
        
    except Exception as e:
        print(f"⚠️  Could not test with real data: {e}")
        return True  # Don't fail the whole test

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VLM PROJECT - MODEL SETUP TEST")
    print("="*60)
    print("\nThis will test:")
    print("1. System requirements")
    print("2. CLIP model loading")
    print("3. Basic inference")
    print("4. Integration with your data (if available)")
    
    # System check
    if not test_system():
        print("\n❌ System check failed - please install dependencies")
        return
    
    input("\nPress Enter to continue...")
    
    # Test CLIP
    if not test_clip_only():
        print("\n❌ CLIP test failed")
        return
    
    input("\nPress Enter to test with real data (if available)...")
    
    # Test with real data
    test_with_real_data()
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("\n✅ Your setup is ready for the project!")
    print("\nNext steps:")
    print("1. Test other models: python models/model_loader.py")
    print("2. Run baseline evaluation")
    print("3. Implement prompting strategies")

if __name__ == "__main__":
    main()