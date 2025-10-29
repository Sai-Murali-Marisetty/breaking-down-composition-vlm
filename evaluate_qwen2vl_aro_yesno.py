"""
Qwen2-VL on ARO — YES/NO protocol (balanced subset friendly)
- Single GPU, fp16, SDPA 'math'
- Strict Yes/No parsing
- Assumes a loader providing iterable examples with:
    {
      "image": <PIL.Image>,
      "caption": <str>,
      "label": 1 for positive match, 0 for negative,
      "subset": <str>  # e.g., VG-Relation, VG-Attribution, Order
    }
- Eval outputs per-subset accuracy + overall.
- Writes results/aro_qwen2vl_yesno_*.csv and summary JSON.
"""

import os, sys, json, traceback, random
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))

import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from load_aro_updated import AROLocalLoader  # <-- adjust if your loader name differs

# ---- CUDA/SDPA safety ----
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
except Exception:
    pass

DTYPE = torch.float16
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")

def sanitize_pil_image(img, max_side=896):
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            m = max(w, h)
            if m > max_side:
                s = max_side / float(m)
                img = img.resize((max(int(w*s), 1), max(int(h*s), 1)), Image.BICUBIC)
        return img
    except Exception:
        return img

def selective_to_device(batch, device):
    out = {}
    allow_gpu = {
        "input_ids", "attention_mask", "position_ids",
        "pixel_values", "pixel_values_videos",
        "decoder_input_ids", "decoder_attention_mask", "labels",
    }
    for k, v in batch.items():
        if torch.is_tensor(v) and k in allow_gpu:
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def parse_yesno(s: str):
    s = (s or "").strip().lower()
    if s.startswith("yes"): return "yes"
    if s.startswith("no"):  return "no"
    if "yes" in s and "no" not in s: return "yes"
    if "no" in s and "yes" not in s: return "no"
    return "unknown"

def ask_yesno(model, processor, device, image, caption, first_fail_flag):
    prompt = (
        "Does this caption correctly describe the image?\n"
        "Answer Yes or No only.\n\n"
        f"Caption: {caption}"
    )
    image = sanitize_pil_image(image)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = selective_to_device(inputs, device)

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    processor.tokenizer.eos_token_id
                    if hasattr(processor, "tokenizer") and processor.tokenizer.eos_token_id is not None
                    else None
                ),
            )
    except Exception:
        if first_fail_flag[0]:
            print("\n--- FIRST FULL STACKTRACE BELOW ---")
            traceback.print_exc()
            print("--- END TRACE ---\n")
            first_fail_flag[0] = False
        raise

    prompt_len = inputs["input_ids"].shape[1]
    output_ids = output_ids[:, prompt_len:]
    resp = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return parse_yesno(resp), resp

def evaluate_qwen2vl_aro_yesno(max_examples=None, seed=1337, balanced_subset=True):
    print("="*60)
    print("QWEN2-VL EVALUATION ON ARO — YES/NO")
    print("="*60)

    # Data
    aro = AROLocalLoader()
    ok = aro.load()
    if not ok or len(aro) == 0:
        raise RuntimeError("ARO not found or empty.")

    # Optional: balanced sub-sampling across subsets/labels
    indices = list(range(len(aro)))
    if balanced_subset:
        # group by (subset, label) and sample evenly
        from collections import defaultdict
        buckets = defaultdict(list)
        for idx in indices:
            ex = aro.get_example(idx)
            buckets[(ex.get("subset","Unknown"), int(ex.get("label",0)))].append(idx)
        random.seed(seed)
        chosen = []
        per_bucket = None
        if max_examples:
            # total buckets
            B = max(1, len(buckets))
            per_bucket = max(1, max_examples // B)
        for k, arr in buckets.items():
            random.shuffle(arr)
            chosen.extend(arr[:per_bucket] if per_bucket else arr)
        indices = chosen
    elif max_examples:
        random.seed(seed); random.shuffle(indices)
        indices = indices[:max_examples]

    print(f"Evaluating N={len(indices)} examples from ARO.")

    # Device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)} | forced dtype={DTYPE}")

    # Model
    print("\n🤖 Loading Qwen/Qwen2-VL-7B-Instruct (single-GPU)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        try:
            model.config.attn_implementation = "eager"
        except Exception:
            pass
    model.to(device, dtype=DTYPE)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True
    )
    try:
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
            ip = processor.image_processor
            if isinstance(ip.size, dict):
                ip.size = {k: min(896, v) for k, v in ip.size.items()}
    except Exception:
        pass

    os.makedirs("results", exist_ok=True)

    correct, errors = 0, 0
    rows = []
    first_fail_flag = [True]

    for i in tqdm(indices, ncols=80):
        ex = aro.get_example(i)
        if ex is None:
            continue
        img = sanitize_pil_image(ex.get("image"))
        cap = ex.get("caption")
        lab = int(ex.get("label", 0))  # 1=positive, 0=negative
        subset = ex.get("subset", "Unknown")

        if img is None or cap is None:
            continue

        try:
            y, raw = ask_yesno(model, processor, device, img, cap, first_fail_flag)  # yes/no/unknown
            pred = 1 if y == "yes" else 0  # unknown counts as 0 (conservative)
            ok = int(pred == lab)
            correct += ok
            rows.append({
                "idx": i, "subset": subset, "label": lab,
                "pred_yes": 1 if y == "yes" else 0,
                "raw": raw, "correct": ok
            })
        except Exception as e:
            errors += 1
            if rows:
                pd.DataFrame(rows).to_csv("results/aro_qwen2vl_yesno_partial.csv", index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if len(rows) % 500 == 0 and len(rows) > 0:
            pd.DataFrame(rows).to_csv("results/aro_qwen2vl_yesno_checkpoint.csv", index=False)
            print(f"\n💾 Checkpoint at {len(rows)} examples")

    N = max(1, len(rows))
    acc = 100.0 * correct / N
    df = pd.DataFrame(rows)

    print("\n" + "="*60)
    print("RESULTS — ARO YES/NO")
    print("="*60)
    print(f"Accuracy: {correct}/{N} ({acc:.2f}%)")
    try:
        by_subset = df.groupby("subset")["correct"].mean().sort_values(ascending=False) * 100
        print("\nPer-subset accuracy (%):")
        print(by_subset.round(1))
    except Exception:
        pass
    if errors:
        print(f"\n⚠ Encountered {errors} errors (see partial/checkpoint CSVs).")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"results/aro_qwen2vl_yesno_{ts}.csv"
    out_json = f"results/aro_qwen2vl_yesno_summary_{ts}.json"
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "model": "Qwen2-VL-7B-Instruct",
            "dtype": str(DTYPE).replace("torch.", ""),
            "num_examples": int(N),
            "correct": int(correct),
            "accuracy": acc,
            "errors": int(errors),
            "timestamp": ts,
            "attention": getattr(getattr(model, "config", object()), "attn_implementation", "unknown"),
            "device_map": "none_single_gpu",
            "protocol": "ARO YES/NO"
        }, f, indent=2)

    print(f"\n💾 Saved: {out_csv}")
    print(f"💾 Saved: {out_json}")
    return acc

if __name__ == "__main__":
    # Example usage: evaluate 2k balanced examples across subsets
    evaluate_qwen2vl_aro_yesno(max_examples=2000, balanced_subset=True)
