"""
Model Loader for VLM Compositional Reasoning Project
Supports: CLIP ViT-B/32, LLaVA-1.5-7B, SmolVLM-Instruct
"""

import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModelForVision2Seq,
    LlavaForConditionalGeneration, AutoTokenizer
)
from PIL import Image
import gc

class ModelLoader:
    """Load and manage VLM models"""
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        
        print(f"🖥️  Device: {self.device}")
        if self.device == "cuda":
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_clip(self):
        """Load CLIP ViT-B/32"""
        print("\n" + "="*60)
        print("Loading CLIP ViT-B/32")
        print("="*60)
        
        try:
            model_name = "openai/clip-vit-base-patch32"
            print(f"Model: {model_name}")
            
            # Load model
            self.models['clip'] = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processors['clip'] = CLIPProcessor.from_pretrained(model_name)
            
            # Set to eval mode
            self.models['clip'].eval()
            
            # Print model info
            num_params = sum(p.numel() for p in self.models['clip'].parameters()) / 1e6
            print(f"✅ CLIP loaded successfully")
            print(f"   Parameters: {num_params:.1f}M")
            print(f"   Memory: ~{num_params * 4 / 1024:.1f} GB (FP32)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading CLIP: {e}")
            return False
    
    def load_llava(self):
        """Load LLaVA-1.5-7B"""
        print("\n" + "="*60)
        print("Loading LLaVA-1.5-7B")
        print("="*60)
        
        try:
            model_name = "llava-hf/llava-1.5-7b-hf"
            print(f"Model: {model_name}")
            print("⚠️  This is a large model (~14GB). Loading may take a few minutes...")
            
            # Load with 16-bit precision to save memory
            self.models['llava'] = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            self.processors['llava'] = AutoProcessor.from_pretrained(model_name)
            
            # Set to eval mode
            self.models['llava'].eval()
            
            # Print model info
            num_params = sum(p.numel() for p in self.models['llava'].parameters()) / 1e9
            print(f"✅ LLaVA loaded successfully")
            print(f"   Parameters: {num_params:.1f}B")
            print(f"   Memory: ~{num_params * 2 / 1024:.1f} GB (FP16)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading LLaVA: {e}")
            return False
    
    def load_smolvlm(self):
        """Load SmolVLM-Instruct"""
        print("\n" + "="*60)
        print("Loading SmolVLM-Instruct")
        print("="*60)
        
        try:
            model_name = "HuggingFaceTB/SmolVLM-Instruct"
            print(f"Model: {model_name}")
            
            # Load with 16-bit precision
            self.models['smolvlm'] = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.processors['smolvlm'] = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set to eval mode
            self.models['smolvlm'].eval()
            
            # Print model info
            num_params = sum(p.numel() for p in self.models['smolvlm'].parameters()) / 1e9
            print(f"✅ SmolVLM loaded successfully")
            print(f"   Parameters: {num_params:.1f}B")
            print(f"   Memory: ~{num_params * 2 / 1024:.1f} GB (FP16)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading SmolVLM: {e}")
            return False
    
    def load_all_models(self):
        """Load all three models"""
        print("\n" + "="*60)
        print("LOADING ALL MODELS")
        print("="*60)
        print("This will take several minutes and use ~20GB RAM")
        print("Loading models sequentially to manage memory...")
        
        success = {}
        
        # Load CLIP first (smallest)
        success['clip'] = self.load_clip()
        
        # Load SmolVLM (medium)
        success['smolvlm'] = self.load_smolvlm()
        
        # Load LLaVA last (largest)
        success['llava'] = self.load_llava()
        
        # Summary
        print("\n" + "="*60)
        print("MODEL LOADING SUMMARY")
        print("="*60)
        
        for model_name, loaded in success.items():
            status = "✅" if loaded else "❌"
            print(f"{status} {model_name.upper()}")
        
        loaded_count = sum(success.values())
        print(f"\n✅ {loaded_count}/3 models loaded successfully")
        
        if self.device == "cuda":
            print(f"\n💾 GPU Memory Usage:")
            print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        return loaded_count == 3
    
    def get_model(self, model_name):
        """Get a loaded model"""
        return self.models.get(model_name)
    
    def get_processor(self, model_name):
        """Get a model's processor"""
        return self.processors.get(model_name)
    
    def unload_model(self, model_name):
        """Unload a specific model to free memory"""
        if model_name in self.models:
            del self.models[model_name]
            del self.processors[model_name]
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print(f"✅ Unloaded {model_name}")
    
    def unload_all(self):
        """Unload all models"""
        self.models.clear()
        self.processors.clear()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("✅ All models unloaded")

def test_model_loading():
    """Test loading each model individually"""
    loader = ModelLoader()
    
    print("\n🧪 Testing individual model loading...")
    print("We'll load one model at a time to test your setup")
    
    # Test CLIP
    input("\nPress Enter to test CLIP loading...")
    if loader.load_clip():
        print("✅ CLIP test passed")
        loader.unload_model('clip')
    
    # Test SmolVLM
    input("\nPress Enter to test SmolVLM loading...")
    if loader.load_smolvlm():
        print("✅ SmolVLM test passed")
        loader.unload_model('smolvlm')
    
    # Test LLaVA
    input("\nPress Enter to test LLaVA loading...")
    if loader.load_llava():
        print("✅ LLaVA test passed")
        loader.unload_model('llava')
    
    print("\n" + "="*60)
    print("✅ All individual tests passed!")
    print("="*60)
    
    # Test loading all together
    load_all = input("\nTest loading all models together? (y/n): ").lower()
    if load_all == 'y':
        loader.load_all_models()

if __name__ == "__main__":
    test_model_loading()