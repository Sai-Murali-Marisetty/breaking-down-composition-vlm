"""
Qwen2-VL on Winoground — A100-safe settings
- Force fp16 (avoid bf16 kernels entirely)
- Force SDPA 'math' path (disable flash/mem-efficient)
- Single GPU only (no sharding)
- Explicit device moves
- RGB + max 896px longest side
- First failure prints full traceback
"""

import os, sys, json, traceback
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))  # local imports

import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from load_winoground_local import WinogroundLocalLoader
from qwen_vl_utils import process_vision_info

# ---------------- CUDA/SDPA safety ----------------
# Use only math SDP kernels; disable flash + mem-efficient
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    # legacy toggles (some builds expose them)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

# Force fp16 even if bf16 is supported (stability)
DTYPE = torch.float16

# Helpful debug (kept on to get precise line if anything fails)
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")

# --------------- helpers -----------------
def sanitize_pil_image(img, max_side=896):
    """RGB + limit longest side to <= max_side."""
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

def to_device(batch, device):
    out = {}
    # tensors we DO move
    allow_gpu = {
        "input_ids", "attention_mask", "position_ids",
        "pixel_values", "pixel_values_videos",
        "decoder_input_ids", "decoder_attention_mask",
        "labels", "past_key_values",
    }
    for k, v in batch.items():
        if torch.is_tensor(v) and k in allow_gpu:
            out[k] = v.to(device, non_blocking=True)
        else:
            # keep everything else (e.g., image_grid_thw, image_sizes) on CPU
            out[k] = v
    return out

def generate_response(model, processor, device, image, prompt, first_fail_flag):
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
    inputs = to_device(inputs, device)

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    processor.tokenizer.eos_token_id
                    if hasattr(processor, "tokenizer") and processor.tokenizer.eos_token_id is not None
                    else None
                ),
            )
    except Exception as e:
        # print detailed trace for the first failure only
        if first_fail_flag[0]:
            print("\n--- FIRST FULL STACKTRACE BELOW ---")
            traceback.print_exc()
            print("--- END TRACE ---\n")
            first_fail_flag[0] = False  # only once
        raise e

    # strip prompt
    prompt_len = inputs["input_ids"].shape[1]
    output_ids = output_ids[:, prompt_len:]
    resp = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # cleanup
    del inputs, output_ids, image_inputs, video_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return resp

def evaluate_qwen2vl_example(model, processor, device, image_0, image_1, caption_0, caption_1, first_fail_flag):
    results = {"text_score": 0, "image_score": 0, "group_score": 0, "responses": {}}
    prompt = (
        "Look at this image. Which caption matches better?\n"
        "A) {cap_a}\n"
        "B) {cap_b}\n\n"
        "Answer with only the letter A or B:"
    )

    r0 = generate_response(model, processor, device, image_0,
                           prompt.format(cap_a=caption_0, cap_b=caption_1),
                           first_fail_flag)
    r1 = generate_response(model, processor, device, image_1,
                           prompt.format(cap_a=caption_0, cap_b=caption_1),
                           first_fail_flag)

    img0_correct = r0.strip().upper().startswith("A")
    img1_correct = r1.strip().upper().startswith("B")

    if img0_correct and img1_correct:
        results["group_score"] = results["text_score"] = results["image_score"] = 1

    results["responses"] = {
        "img0_response": r0, "img1_response": r1,
        "img0_correct": img0_correct, "img1_correct": img1_correct,
    }
    return results

# --------------- main -----------------
def evaluate_qwen2vl(num_examples=50):
    print("=" * 60)
    print("QWEN2-VL EVALUATION ON WINOGROUND (fp16 + SDPA-math, single GPU)")
    print("=" * 60)

    # Dataset
    print("\n📊 Loading Winoground...")
    wg = WinogroundLocalLoader()
    ok = wg.load()
    if not ok or len(wg) == 0:
        raise RuntimeError("Winoground not found or empty.")
    num_examples = min(num_examples, len(wg))

    # Device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — this file assumes GPU.")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)} | forced dtype={DTYPE}")

    # Model — single GPU, force eager/sdpa math
    print("\n🤖 Loading Qwen/Qwen2-VL-7B-Instruct (single-GPU)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,   # no sharding
    )
    # Prefer SDPA (non-flash) or eager; some stacks accept 'sdpa'
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
    # Optional: bound image processor size if available
    try:
        if hasattr(processor, "image_processor"):
            ip = processor.image_processor
            if hasattr(ip, "size"):
                # set a conservative size
                if isinstance(ip.size, dict):
                    ip.size = {k: min(896, v) for k, v in ip.size.items()}
    except Exception:
        pass

    os.makedirs("results", exist_ok=True)

    print(f"\n🔄 Evaluating {num_examples} examples...")
    correct, errors = 0, 0
    details = []
    first_fail_flag = [True]  # mutable flag so we only print full trace once

    for i in tqdm(range(num_examples), ncols=80):
        ex = wg.get_example(i)
        if ex is None:
            print(f"\n⚠ Example {i} missing; skipping.")
            continue

        img0 = sanitize_pil_image(ex.get("image_0"))
        img1 = sanitize_pil_image(ex.get("image_1"))
        cap0, cap1 = ex.get("caption_0"), ex.get("caption_1")
        if img0 is None or img1 is None or cap0 is None or cap1 is None:
            print(f"\n⚠ Example {i} missing fields; skipping.")
            continue

        try:
            res = evaluate_qwen2vl_example(
                model, processor, device, img0, img1, cap0, cap1, first_fail_flag
            )
            correct += res["group_score"]
            details.append({
                "id": i, "tag": ex.get("tag"),
                "caption_0": cap0, "caption_1": cap1,
                "group_score": res["group_score"],
                "img0_response": res["responses"]["img0_response"],
                "img1_response": res["responses"]["img1_response"],
                "img0_correct": res["responses"]["img0_correct"],
                "img1_correct": res["responses"]["img1_correct"],
            })
        except Exception as e:
            errors += 1
            print(f"\nError on example {i}: {e}")
            if details:
                pd.DataFrame(details).to_csv(f"results/qwen2vl_partial_{i}.csv", index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if (i + 1) % 25 == 0:
            pd.DataFrame(details).to_csv(f"results/qwen2vl_checkpoint_{i+1}.csv", index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\n💾 Checkpoint saved at {i+1} examples")

    acc = 100.0 * correct / max(1, num_examples)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {correct}/{num_examples} ({acc:.2f}%)")
    if errors:
        print(f"⚠ Encountered {errors} errors (see partials).")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(details).to_csv(f"results/qwen2vl_winoground_{ts}.csv", index=False)
    with open(f"results/qwen2vl_summary_{ts}.json", "w") as f:
        json.dump({
            "model": "Qwen2-VL-7B-Instruct",
            "dtype": str(DTYPE).replace("torch.", ""),
            "num_examples": num_examples,
            "correct": int(correct),
            "accuracy": acc,
            "errors": int(errors),
            "timestamp": ts,
            "attention": getattr(getattr(model, "config", object()), "attn_implementation", "unknown"),
            "device_map": "none_single_gpu"
        }, f, indent=2)

    print("\n💾 Results saved to results/")
    return acc

if __name__ == "__main__":
    evaluate_qwen2vl(num_examples=400)
