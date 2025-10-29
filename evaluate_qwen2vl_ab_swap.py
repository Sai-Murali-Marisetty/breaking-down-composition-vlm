"""
Qwen2-VL on Winoground — A/B SWAP evaluation
- Same stable path as your successful run:
  * single GPU (no sharding)
  * fp16
  * SDPA 'math' attention (no flash/mem-efficient)
  * selective device move (keep meta/grid tensors on CPU)
  * RGB images, max 896px longest side
- Difference: captions are SWAPPED (A<->B) before querying the model.
- Output files are suffixed with `_ab_swap`.
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
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    # Some stacks expose legacy toggles:
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
except Exception:
    pass

DTYPE = torch.float16  # force fp16 for stability on A100

# Helpful debug for precise traces if anything goes wrong
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")


# ---------------- helpers ----------------
def sanitize_pil_image(img, max_side=896):
    """Ensure PIL RGB and cap longest side to <= max_side."""
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            m = max(w, h)
            if m > max_side:
                s = max_side / float(m)
                img = img.resize((max(int(w * s), 1), max(int(h * s), 1)), Image.BICUBIC)
        return img
    except Exception:
        return img


def selective_to_device(batch, device):
    """Move only model input tensors; keep meta/grid tensors on CPU."""
    out = {}
    allow_gpu = {
        "input_ids", "attention_mask", "position_ids",
        "pixel_values", "pixel_values_videos",
        "decoder_input_ids", "decoder_attention_mask",
        "labels",
    }
    for k, v in batch.items():
        if torch.is_tensor(v) and k in allow_gpu:
            out[k] = v.to(device, non_blocking=True)
        else:
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
    inputs = selective_to_device(inputs, device)

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

    del inputs, output_ids, image_inputs, video_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return resp


def evaluate_example_ab_swap(model, processor, device, image_0, image_1, caption_0, caption_1, first_fail_flag):
    """
    Winoground scoring logic expects:
      - For the ORIGINAL setting: img0 -> A (cap0), img1 -> B (cap1)
    In A/B SWAP, we swap the CAPTIONS shown to the model:
      - The model sees A=cap1, B=cap0 for BOTH images.
    We still evaluate correctness relative to the *original* task definition.
    That means after swapping prompts:
      - img0 is correct if the model now chooses **B** (since B=original A)
      - img1 is correct if the model now chooses **A** (since A=original B)
    """
    prompt_tmpl = (
        "Look at this image. Which caption matches better?\n"
        "A) {cap_a}\n"
        "B) {cap_b}\n\n"
        "Answer with only the letter A or B:"
    )

    # SWAP: present cap1 as A, cap0 as B
    swapped_A = caption_1
    swapped_B = caption_0

    r0 = generate_response(
        model, processor, device, image_0,
        prompt_tmpl.format(cap_a=swapped_A, cap_b=swapped_B),
        first_fail_flag
    )
    r1 = generate_response(
        model, processor, device, image_1,
        prompt_tmpl.format(cap_a=swapped_A, cap_b=swapped_B),
        first_fail_flag
    )

    # Under SWAP, correctness flips:
    # img0 should now answer 'B' (because B==original A)
    # img1 should now answer 'A' (because A==original B)
    img0_correct = r0.strip().upper().startswith("B")
    img1_correct = r1.strip().upper().startswith("A")

    group = 1 if (img0_correct and img1_correct) else 0
    return {
        "group_score": group,
        "responses": {
            "img0_response": r0,
            "img1_response": r1,
            "img0_correct": img0_correct,
            "img1_correct": img1_correct,
        }
    }


# ---------------- main ----------------
def evaluate_qwen2vl_ab_swap(num_examples=400):
    print("=" * 60)
    print("QWEN2-VL EVALUATION ON WINOGROUND — A/B SWAP")
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

    # Model — single GPU, SDPA/eager
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
    # Conservative image size if available
    try:
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
            ip = processor.image_processor
            if isinstance(ip.size, dict):
                ip.size = {k: min(896, v) for k, v in ip.size.items()}
    except Exception:
        pass

    os.makedirs("results", exist_ok=True)

    print(f"\n🔄 Evaluating {num_examples} examples (A/B swapped captions)...")
    correct, errors = 0, 0
    details = []
    first_fail_flag = [True]

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
            res = evaluate_example_ab_swap(
                model, processor, device, img0, img1, cap0, cap1, first_fail_flag
            )
            correct += res["group_score"]
            details.append({
                "id": i,
                "tag": ex.get("tag"),
                "caption_0": cap0,      # original
                "caption_1": cap1,      # original
                "caption_A_shown": cap1,  # swapped A
                "caption_B_shown": cap0,  # swapped B
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
                pd.DataFrame(details).to_csv(f"results/qwen2vl_ab_swap_partial_{i}.csv", index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if (i + 1) % 25 == 0:
            pd.DataFrame(details).to_csv(f"results/qwen2vl_ab_swap_checkpoint_{i+1}.csv", index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\n💾 Checkpoint saved at {i+1} examples")

    acc = 100.0 * correct / max(1, num_examples)
    print("\n" + "=" * 60)
    print("RESULTS — A/B SWAP")
    print("=" * 60)
    print(f"Accuracy: {correct}/{num_examples} ({acc:.2f}%)")
    if errors:
        print(f"⚠ Encountered {errors} errors (see partials).")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(details).to_csv(f"results/qwen2vl_winoground_ab_swap_{ts}.csv", index=False)
    with open(f"results/qwen2vl_summary_ab_swap_{ts}.json", "w") as f:
        json.dump({
            "model": "Qwen2-VL-7B-Instruct",
            "dtype": str(DTYPE).replace("torch.", ""),
            "num_examples": num_examples,
            "correct": int(correct),
            "accuracy": acc,
            "errors": int(errors),
            "timestamp": ts,
            "attention": getattr(getattr(model, "config", object()), "attn_implementation", "unknown"),
            "device_map": "none_single_gpu",
            "protocol": "A/B swap (captions swapped)"
        }, f, indent=2)

    print("\n💾 Results saved to results/")
    return acc


if __name__ == "__main__":
    evaluate_qwen2vl_ab_swap(num_examples=400)
