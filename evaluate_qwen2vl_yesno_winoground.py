"""
Qwen2-VL on Winoground — YES/NO protocol
- Single GPU (no sharding)
- fp16 + SDPA 'math' (no flash/mem-efficient)
- Selective device move (keep meta/grid tensors on CPU)
- RGB images, max 896px longest side
- Scoring:
  For image_0: caption_0 => Yes AND caption_1 => No
  For image_1: caption_1 => Yes AND caption_0 => No
  Group = 1 if both images satisfy the condition; else 0.
- Outputs: results/qwen2vl_winoground_yesno_*.csv + summary JSON
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

DTYPE = torch.float16  # force fp16 on A100 for stability

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
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    # ultra-strict fallback: check tokens
    if "yes" in s and "no" not in s:
        return "yes"
    if "no" in s and "yes" not in s:
        return "no"
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

def evaluate_qwen2vl_yesno(num_examples=400):
    print("="*60)
    print("QWEN2-VL EVALUATION ON WINOGROUND — YES/NO")
    print("="*60)

    # Dataset
    wg = WinogroundLocalLoader()
    ok = wg.load()
    if not ok or len(wg) == 0:
        raise RuntimeError("Winoground not found or empty.")
    num_examples = min(num_examples, len(wg))

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

    print(f"\n🔄 Evaluating {num_examples} examples (YES/NO protocol)...")
    correct, errors = 0, 0
    details = []
    first_fail_flag = [True]

    for i in tqdm(range(num_examples), ncols=80):
        ex = wg.get_example(i)
        if ex is None:
            continue
        img0, img1 = sanitize_pil_image(ex.get("image_0")), sanitize_pil_image(ex.get("image_1"))
        cap0, cap1 = ex.get("caption_0"), ex.get("caption_1")
        if img0 is None or img1 is None or cap0 is None or cap1 is None:
            continue

        try:
            y00, raw00 = ask_yesno(model, processor, device, img0, cap0, first_fail_flag)  # should be Yes
            y01, raw01 = ask_yesno(model, processor, device, img0, cap1, first_fail_flag)  # should be No
            y10, raw10 = ask_yesno(model, processor, device, img1, cap0, first_fail_flag)  # should be No
            y11, raw11 = ask_yesno(model, processor, device, img1, cap1, first_fail_flag)  # should be Yes

            img0_ok = (y00 == "yes") and (y01 == "no")
            img1_ok = (y11 == "yes") and (y10 == "no")
            group = 1 if (img0_ok and img1_ok) else 0
            correct += group

            details.append({
                "id": i,
                "tag": ex.get("tag"),
                "caption_0": cap0, "caption_1": cap1,
                "img0_cap0": y00, "img0_cap1": y01,
                "img1_cap0": y10, "img1_cap1": y11,
                "img0_raw": raw00, "img0_raw_alt": raw01,
                "img1_raw": raw11, "img1_raw_alt": raw10,
                "group_score": group,
                "img0_correct": img0_ok, "img1_correct": img1_ok,
            })
        except Exception as e:
            errors += 1
            if details:
                pd.DataFrame(details).to_csv(f"results/qwen2vl_yesno_partial_{i}.csv", index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if (i + 1) % 25 == 0:
            pd.DataFrame(details).to_csv(f"results/qwen2vl_yesno_checkpoint_{i+1}.csv", index=False)
            print(f"\n💾 Checkpoint saved at {i+1} examples")

    acc = 100.0 * correct / max(1, num_examples)
    print("\n" + "="*60)
    print("RESULTS — YES/NO")
    print("="*60)
    print(f"Accuracy: {correct}/{num_examples} ({acc:.2f}%)")
    if errors:
        print(f"⚠ Encountered {errors} errors (see partials).")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(details).to_csv(f"results/qwen2vl_winoground_yesno_{ts}.csv", index=False)
    with open(f"results/qwen2vl_summary_yesno_{ts}.json", "w") as f:
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
            "protocol": "Winoground YES/NO"
        }, f, indent=2)

    print("\n💾 Results saved to results/")
    return acc

if __name__ == "__main__":
    evaluate_qwen2vl_yesno(num_examples=400)
