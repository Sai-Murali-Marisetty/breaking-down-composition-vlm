# Breaking Down Composition: Analyzing Failure Modes in Vision-Language Models

A systematic evaluation of compositional reasoning capabilities in Vision-Language Models (VLMs) and the effectiveness of prompt-based interventions.

## 📋 Project Overview

### Research Questions

1. What specific compositional reasoning capabilities do modern VLMs lack?
2. Can simple prompting strategies improve their performance without retraining?

### Key Findings

- ✅ **CLIP baseline**: 31.25% group score on Winoground (matches published benchmarks)
- ✅ **Error taxonomy**: Identified 5 major failure categories
- ✅ **Prompting effectiveness**: Zero improvement for CLIP (embedding model)
- 🔄 **In progress**: Testing generative models (LLaVA, SmolVLM)

---

## 🗂️ Repository Structure

```
breaking-down-composition-vlm/
├── data/                          # Datasets (not included in git)
│   ├── winoground/               # 400 examples
│   └── aro/                      # 52K examples
│
├── models/                        # Model infrastructure
│   ├── model_loader.py           # Load CLIP, LLaVA, SmolVLM
│   └── inference_engine.py       # Inference wrapper
│
├── scripts/                       # Evaluation scripts
│   ├── baseline_winoground_clip.py
│   └── baseline_aro_clip.py
│
├── results/                       # Evaluation results
│   ├── *.json                    # Summary metrics
│   └── *.csv                     # Detailed results
│
├── Dataset loaders
├── load_winoground_local.py      # Winoground dataset loader
├── load_aro_updated.py           # ARO dataset loader
├── verify_aro_data.py            # Data verification
│
├── Evaluation scripts
├── baseline_simple.py            # Simple baseline evaluation
├── evaluate_prompts.py           # Prompting strategies evaluation
├── evaluate_llava_baseline.py    # LLaVA evaluation
│
├── Analysis
├── analyze_results.py            # Results analysis
├── prompting_strategies.py       # Prompt templates
│
├── Testing
├── simple_test.py                # Quick model test
├── test_llava_setup.py           # LLaVA setup verification
├── test_models_setup.py          # Full model test
├── test_with_winoground.py       # Winoground test
│
├── Configuration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd breaking-down-composition-vlm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

**Winoground** (400 examples):
- Visit: https://huggingface.co/datasets/facebook/winoground
- Accept terms and download to `data/winoground/`

**ARO** (52K examples):
- Clone: https://github.com/mertyg/vision-language-models-are-bows
- Copy data files to `data/aro/`

### 3. Run Baseline Evaluation

```bash
# Quick test (5 examples)
python test_with_winoground.py

# Full evaluation
python baseline_simple.py
```

---

## 📊 Results Summary

### CLIP ViT-B/32 Performance

**Winoground (400 examples)**:
- **Group Score**: 31.25%
- **Text Score**: 64.50%
- **Image Score**: 31.25%

**ARO (2,000 examples)**:
- **VG-Relation**: 54.40%
- **VG-Attribution**: 56.90%
- **Overall**: 55.65%

### Prompting Strategy Results (CLIP)

| Strategy | Group Score | Change |
|----------|-------------|--------|
| Zero-Shot (Baseline) | 31.25% | - |
| Explicit Decomposition | 31.25% | 0.00% |
| Chain-of-Thought | 31.25% | 0.00% |
| Contrastive | 31.25% | 0.00% |

**Finding**: Prompting has **zero effect** on embedding-based models like CLIP.

### Error Taxonomy

1. **Attribute Confusion** (High severity)
   - Cannot distinguish colors, sizes, ages
   
2. **Relation/Role Confusion** (High severity)
   - Fails to identify agent vs patient roles
   
3. **Word Order Insensitivity** (Medium severity)
   - Ignores word sequence changes
   
4. **Negation Failures** (High severity)
   - Cannot process negations correctly
   
5. **Spatial Reasoning** (Medium severity)
   - Struggles with spatial prepositions

---

## 🔬 Methodology

### Models Evaluated

1. **CLIP ViT-B/32** (151M params) ✅ Complete
   - Embedding-based model
   - Zero-shot baseline: 31.25%
   
2. **LLaVA-1.5-7B** (7B params) 🔄 In Progress
   - Generative model
   - Expected to benefit from prompting
   
3. **SmolVLM-Instruct** (2B params) ⏳ Planned
   - Small efficient model

### Evaluation Datasets

1. **Winoground** (400 examples)
   - Minimal pairs with identical words, different order
   - Tests: text score, image score, group score
   
2. **ARO** (52K examples)
   - VG-Relation: Tests relational understanding
   - VG-Attribution: Tests attribute recognition

### Prompting Strategies

1. **Zero-Shot Baseline**: Standard caption
2. **Explicit Decomposition**: Break into objects → attributes → relationships
3. **Chain-of-Thought**: Step-by-step reasoning
4. **Contrastive**: Explicit comparison between options

---

## 📈 Key Insights

### 1. CLIP's Limitations

- **Strong text encoder** (64.5% text score)
- **Weak image-text alignment** (31.25% image score)
- **Bottleneck**: Image understanding, not caption processing

### 2. Prompting Ineffectiveness (CLIP)

- Zero improvement across all strategies
- Embedding models cannot process reasoning instructions
- Architecture fundamentally limits compositional reasoning

### 3. Error Patterns

- Worst performance on negation and relations
- Better on simple attributes and colors
- Significant variation by compositional tag type

---

## 🛠️ Technical Details

### System Requirements

- **RAM**: 24GB+ recommended
- **GPU**: CUDA-capable (25GB+ for LLaVA)
- **Python**: 3.8+

### Key Dependencies

- PyTorch 2.0+
- Transformers 4.35+
- CLIP (OpenAI)
- Pillow, pandas, tqdm

### Memory Usage

| Model | Memory (FP16) | Status |
|-------|---------------|--------|
| CLIP | ~0.6 GB | ✅ Tested |
| LLaVA | ~14 GB | 🔄 Testing |
| SmolVLM | ~4 GB | ⏳ Planned |

---

## 📝 Usage Examples

### Evaluate CLIP on Winoground

```python
from models.model_loader import ModelLoader
from load_winoground_local import WinogroundLocalLoader

# Load dataset
loader = WinogroundLocalLoader()
loader.load()

# Load model
model_loader = ModelLoader()
model_loader.load_clip()

# Run evaluation
python baseline_simple.py
```

### Test Prompting Strategies

```python
python evaluate_prompts.py
# Choose: 1 (Quick test) or 2 (Full evaluation)
```

### Analyze Results

```python
python analyze_results.py
# Generates tag analysis and error taxonomy
```

---

## 📊 Results Files

All results saved to `results/` directory:

- `*_summary_*.json`: Overall metrics
- `*_detailed_*.csv`: Per-example results  
- `*_by_tag_*.csv`: Analysis by compositional tag
- `error_taxonomy_draft.json`: Failure category taxonomy

---

## 🎓 Academic Context

### Related Work

- **Winoground** (Thrush et al., 2022): Benchmark for visio-linguistic compositionality
- **ARO** (Yuksekgonul et al., 2023): Vision-language models as bags-of-words
- **CLIP** (Radford et al., 2021): Contrastive language-image pre-training
- **LLaVA** (Liu et al., 2023): Visual instruction tuning

### Contributions

1. Systematic evaluation across models and strategies
2. Comprehensive error taxonomy (5 categories)
3. Quantitative evidence of prompting ineffectiveness for embedding models
4. Cross-architecture comparison (embedding vs generative)

---

## 📅 Project Timeline

- **Weeks 1-2**: Data preparation ✅
- **Week 3**: Baseline evaluation ✅
- **Week 4**: Prompting strategies ✅ (CLIP complete)
- **Week 5**: Midterm report 🔄
- **Weeks 6-8**: Full experiments (LLaVA, SmolVLM)
- **Weeks 9-10**: Final paper

---

**Last Updated**: October 26, 2025  
**Status**: Week 4 - Baseline evaluation complete, LLaVA testing in progress