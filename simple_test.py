"""
Simple test script to verify model setup
Fixed import paths for your directory structure
"""

import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

def test_imports():
    """Test that all imports work"""
    print("="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ PIL/Pillow")
    except ImportError as e:
        print(f"❌ PIL: {e}")
        return False
    
    try:
        from model_loader import ModelLoader
        print(f"✅ ModelLoader module")
    except ImportError as e:
        print(f"❌ ModelLoader: {e}")
        print(f"   Make sure model_loader.py is in models/ directory")
        return False
    
    try:
        from inference_engine import VLMInference
        print(f"✅ VLMInference module")
    except ImportError as e:
        print(f"❌ VLMInference: {e}")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    import torch
    
    print("\n" + "="*60)
    print("CUDA CHECK")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Available")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total Memory: {total_mem:.1f} GB")
        
        if total_mem >= 20:
            print(f"   ✅ Sufficient for all models!")
        else:
            print(f"   ⚠️  May need to run models one at a time")
    else:
        print(f"⚠️  CUDA not available - will use CPU (much slower)")
        print(f"   This is OK for testing, but evaluation will be slow")
    
    return True

def test_clip_loading():
    """Test CLIP model loading"""
    from model_loader import ModelLoader
    import torch
    
    print("\n" + "="*60)
    print("TESTING CLIP LOADING")
    print("="*60)
    print("This will download CLIP model (~400MB) on first run...")
    
    try:
        loader = ModelLoader()
        
        print("\nLoading CLIP...")
        success = loader.load_clip()
        
        if success:
            print("\n✅ CLIP loaded successfully!")
            
            # Check memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"   GPU Memory Used: {allocated:.2f} GB")
            
            # Clean up
            loader.unload_all()
            print("   Cleaned up memory")
            
            return True
        else:
            print("❌ Failed to load CLIP")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_inference():
    """Test basic inference with dummy image"""
    from model_loader import ModelLoader
    from inference_engine import VLMInference
    from PIL import Image
    
    print("\n" + "="*60)
    print("TESTING BASIC INFERENCE")
    print("="*60)
    
    try:
        # Load model
        loader = ModelLoader()
        if not loader.load_clip():
            return False
        
        # Create inference engine
        inference = VLMInference(
            model=loader.get_model('clip'),
            processor=loader.get_processor('clip'),
            model_type='clip',
            device=loader.device
        )
        
        # Create dummy image
        print("\nCreating test image...")
        image = Image.new('RGB', (224, 224), color='blue')
        
        # Test captions
        captions = [
            "a blue square",
            "a red circle",
            "a cat"
        ]
        
        print("Computing similarities...")
        scores = inference.compute_similarity_clip(image, captions)
        
        print("\n📊 Results:")
        for caption, score in zip(captions, scores):
            print(f"   {caption:20s}: {score:.4f}")
        
        # The blue square should score highest
        if scores[0] > scores[1] and scores[0] > scores[2]:
            print("\n✅ Inference working correctly!")
            print("   (Blue square scored highest for blue image)")
        else:
            print("\n⚠️  Scores unexpected, but inference ran")
        
        # Clean up
        loader.unload_all()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VLM PROJECT - SIMPLE MODEL TEST")
    print("="*60)
    
    print("\nThis will test:")
    print("1. Python imports")
    print("2. CUDA availability") 
    print("3. CLIP model loading")
    print("4. Basic inference")
    
    input("\nPress Enter to start...")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt --break-system-packages")
        return
    
    # Test CUDA
    test_cuda()
    
    input("\nPress Enter to test CLIP loading (will download ~400MB)...")
    
    # Test CLIP
    if not test_clip_loading():
        print("\n❌ CLIP loading failed")
        return
    
    input("\nPress Enter to test inference...")
    
    # Test inference
    if not test_basic_inference():
        print("\n❌ Inference test failed")
        return
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    
    print("\n✅ Your setup is working!")
    print("\nNext steps:")
    print("1. Test with real data: python test_with_winoground.py")
    print("2. Test other models: cd models && python model_loader.py")
    print("3. Run baseline evaluation")

if __name__ == "__main__":
    main()