"""
Inference Engine for VLM models
Handles image-text matching and text generation
"""

import torch
from PIL import Image
from typing import List, Dict
import numpy as np

class VLMInference:
    """Run inference with VLM models"""
    
    def __init__(self, model, processor, model_type, device="cuda"):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.device = device
    
    @torch.no_grad()
    def compute_similarity_clip(self, image, texts):
        """
        Compute image-text similarity for CLIP
        Returns: similarity scores for each text
        """
        # Process inputs
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get embeddings
        outputs = self.model(**inputs)
        
        # Compute similarity (image vs each text)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        return probs.cpu().numpy()[0]
    
    @torch.no_grad()
    def generate_caption_llava(self, image, prompt="Describe this image."):
        """
        Generate caption with LLaVA
        """
        # Format prompt for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        prompt_text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
        
        # Decode
        output_text = self.processor.decode(
            output_ids[0], 
            skip_special_tokens=True
        )
        
        return output_text
    
    @torch.no_grad()
    def generate_caption_smolvlm(self, image, prompt="Describe this image."):
        """
        Generate caption with SmolVLM
        """
        # Format messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        prompt_text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
        
        # Decode
        output_text = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        return output_text
    
    def evaluate_winoground_example(self, image_0, image_1, caption_0, caption_1):
        """
        Evaluate a Winoground example
        Returns scores for text_score, image_score, group_score
        """
        if self.model_type == "clip":
            # Get similarity scores
            scores_img0 = self.compute_similarity_clip(image_0, [caption_0, caption_1])
            scores_img1 = self.compute_similarity_clip(image_1, [caption_0, caption_1])
            
            # Text score: C0 matches I0 better than I1, AND C1 matches I1 better than I0
            text_correct_0 = scores_img0[0] > scores_img1[0]  # C0 prefers I0
            text_correct_1 = scores_img1[1] > scores_img0[1]  # C1 prefers I1
            text_score = 1.0 if (text_correct_0 and text_correct_1) else 0.0
            
            # Image score: I0 matches C0 better than C1, AND I1 matches C1 better than C0
            image_correct_0 = scores_img0[0] > scores_img0[1]  # I0 prefers C0
            image_correct_1 = scores_img1[1] > scores_img1[0]  # I1 prefers C1
            image_score = 1.0 if (image_correct_0 and image_correct_1) else 0.0
            
            # Group score: both text and image correct
            group_score = 1.0 if (text_score == 1.0 and image_score == 1.0) else 0.0
            
            return {
                "text_score": text_score,
                "image_score": image_score,
                "group_score": group_score,
                "scores_img0": scores_img0.tolist(),
                "scores_img1": scores_img1.tolist()
            }
        
        else:
            # For generative models (LLaVA, SmolVLM)
            # We'll implement a different evaluation strategy
            raise NotImplementedError(f"Winoground eval not yet implemented for {self.model_type}")
    
    def evaluate_aro_example(self, image, true_caption, false_caption):
        """
        Evaluate an ARO example (binary classification)
        Returns: 1 if true_caption scores higher, 0 otherwise
        """
        if self.model_type == "clip":
            scores = self.compute_similarity_clip(image, [true_caption, false_caption])
            correct = 1 if scores[0] > scores[1] else 0
            return {
                "correct": correct,
                "true_score": float(scores[0]),
                "false_score": float(scores[1])
            }
        else:
            raise NotImplementedError(f"ARO eval not yet implemented for {self.model_type}")

def test_inference():
    """Test inference with a sample image"""
    import sys
    import os
    
    # Add current directory to path if running from models/
    sys.path.insert(0, os.path.dirname(__file__))
    
    from model_loader import ModelLoader
    import requests
    from io import BytesIO
    
    print("🧪 Testing inference...")
    
    # Load CLIP (smallest model for testing)
    loader = ModelLoader()
    if not loader.load_clip():
        print("❌ Failed to load model")
        return
    
    # Create inference engine
    inference = VLMInference(
        model=loader.get_model('clip'),
        processor=loader.get_processor('clip'),
        model_type='clip',
        device=loader.device
    )
    
    # Load a test image
    print("\n📸 Loading test image...")
    url = "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400"
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        print("✅ Image loaded")
    except:
        print("⚠️  Using placeholder - network may be restricted")
        image = Image.new('RGB', (224, 224), color='red')
    
    # Test similarity
    print("\n🔍 Testing image-text similarity...")
    captions = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird"
    ]
    
    scores = inference.compute_similarity_clip(image, captions)
    print("Similarity scores:")
    for caption, score in zip(captions, scores):
        print(f"  {caption:30s}: {score:.4f}")
    
    print("\n✅ Inference test complete!")

if __name__ == "__main__":
    test_inference()