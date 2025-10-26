"""
Prompting Strategies for Compositional Reasoning
Four approaches based on error taxonomy
"""

class PromptingStrategies:
    """
    Four prompting strategies from your proposal:
    1. Zero-shot baseline (already done!)
    2. Explicit decomposition
    3. Chain-of-thought reasoning
    4. Contrastive prompting
    """
    
    @staticmethod
    def zero_shot_baseline():
        """
        Strategy 1: Zero-shot (Default)
        Just the caption as-is, no modifications
        """
        return {
            "name": "Zero-Shot Baseline",
            "description": "Standard CLIP inference with no prompt engineering",
            "template": "{caption}",
            "example": "a red car near a blue house"
        }
    
    @staticmethod
    def explicit_decomposition():
        """
        Strategy 2: Explicit Decomposition
        Break down into: objects → attributes → relationships
        
        Addresses:
        - Attribute Confusion
        - Relation/Role Confusion
        """
        return {
            "name": "Explicit Decomposition",
            "description": "Break caption into objects, attributes, and relationships",
            "template": "Analyze this image by identifying: 1) Objects present, 2) Their attributes (color, size, etc.), 3) Their relationships. Caption: {caption}",
            "example": "Analyze this image by identifying: 1) Objects present, 2) Their attributes (color, size, etc.), 3) Their relationships. Caption: a red car near a blue house"
        }
    
    @staticmethod
    def chain_of_thought():
        """
        Strategy 3: Chain-of-Thought
        Encourage step-by-step reasoning
        
        Addresses:
        - Word Order Insensitivity
        - Negation Failures
        - Spatial Reasoning
        """
        return {
            "name": "Chain-of-Thought",
            "description": "Encourage systematic reasoning about image-caption match",
            "template": "Let's think step by step about this image. Does it match this caption: {caption}? Consider: What objects are present? What are their attributes? What relationships exist between them?",
            "example": "Let's think step by step about this image. Does it match this caption: a red car near a blue house? Consider: What objects are present? What are their attributes? What relationships exist between them?"
        }
    
    @staticmethod
    def contrastive_prompting():
        """
        Strategy 4: Contrastive Prompting
        Explicitly compare two options
        
        Addresses:
        - All failure types (forces explicit comparison)
        """
        return {
            "name": "Contrastive Prompting",
            "description": "Explicitly compare two caption options",
            "template": "Compare these two captions for this image: Caption A: {caption_a}. Caption B: {caption_b}. Which caption better describes the image? Pay close attention to word order and relationships.",
            "example": "Compare these two captions for this image: Caption A: a red car near a blue house. Caption B: a blue car near a red house. Which caption better describes the image? Pay close attention to word order and relationships."
        }
    
    @staticmethod
    def get_all_strategies():
        """Get all prompting strategies"""
        return [
            PromptingStrategies.zero_shot_baseline(),
            PromptingStrategies.explicit_decomposition(),
            PromptingStrategies.chain_of_thought(),
            PromptingStrategies.contrastive_prompting()
        ]

# Prompt templates for different failure categories
CATEGORY_SPECIFIC_PROMPTS = {
    "Attribute Confusion": {
        "focus": "Pay special attention to attributes like color, size, and age",
        "template": "In this image, carefully identify the attributes (colors, sizes, ages) of each object. Caption: {caption}"
    },
    
    "Relation/Role Confusion": {
        "focus": "Identify who is doing what to whom",
        "template": "In this image, identify the agent (who is doing the action) and the patient (who receives the action). Caption: {caption}"
    },
    
    "Word Order Insensitivity": {
        "focus": "Word order matters - analyze the sequence carefully",
        "template": "Pay careful attention to the order of words in this caption. The sequence matters. Caption: {caption}"
    },
    
    "Negation Failures": {
        "focus": "Look for negations (not, without, un-)",
        "template": "Carefully check for any negations or negative words in the caption. What is present? What is absent? Caption: {caption}"
    },
    
    "Spatial Reasoning": {
        "focus": "Identify spatial relationships (on, in, above, below, near)",
        "template": "Identify the spatial relationships between objects. Where is each object located relative to others? Caption: {caption}"
    }
}

def format_prompt(strategy, caption, caption_b=None, category=None):
    """
    Format a prompt using a strategy
    
    Args:
        strategy: Strategy dict from PromptingStrategies
        caption: The caption text
        caption_b: Optional second caption for contrastive prompting
        category: Optional failure category for category-specific prompts
    
    Returns:
        Formatted prompt string
    """
    template = strategy["template"]
    
    # Format basic template
    if "{caption_a}" in template and "{caption_b}" in template:
        # Contrastive - needs both captions
        if caption_b:
            return template.format(caption_a=caption, caption_b=caption_b)
        else:
            # Fallback if no second caption
            return template.format(caption_a=caption, caption_b="[alternative caption]")
    else:
        # Standard single caption
        prompt = template.format(caption=caption)
        
        # Add category-specific guidance if provided
        if category and category in CATEGORY_SPECIFIC_PROMPTS:
            category_prompt = CATEGORY_SPECIFIC_PROMPTS[category]
            prompt = f"{category_prompt['focus']}. {prompt}"
        
        return prompt

def demonstrate_prompts():
    """Demonstrate all prompting strategies"""
    
    print("="*60)
    print("PROMPTING STRATEGIES DEMONSTRATION")
    print("="*60)
    
    # Example Winoground pair
    caption_0 = "an old person kisses a young person"
    caption_1 = "a young person kisses an old person"
    
    print(f"\nExample Winoground Pair:")
    print(f"  Caption 0: {caption_0}")
    print(f"  Caption 1: {caption_1}")
    print(f"  Tag: Adjective-Age (Attribute Confusion)")
    
    strategies = PromptingStrategies.get_all_strategies()
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*60}")
        print(f"Strategy {i}: {strategy['name']}")
        print(f"{'='*60}")
        print(f"Description: {strategy['description']}")
        print(f"\nPrompted Caption 0:")
        
        if strategy['name'] == "Contrastive Prompting":
            print(f"  {format_prompt(strategy, caption_0, caption_1)}")
        else:
            print(f"  {format_prompt(strategy, caption_0, category='Attribute Confusion')}")
    
    print("\n" + "="*60)
    print("Note: These prompts are for GENERATIVE models (LLaVA, SmolVLM)")
    print("CLIP uses embeddings, so we'll adapt the approach")
    print("="*60)

if __name__ == "__main__":
    demonstrate_prompts()
    
    print("\n💡 Implementation Notes:")
    print("="*60)
    print("1. CLIP (embeddings): Test if longer prompts change similarity scores")
    print("2. LLaVA/SmolVLM (generative): Use prompts to guide reasoning")
    print("3. Contrastive: Works best with generative models")
    print("4. Decomposition: May help focus on specific attributes")