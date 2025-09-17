#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Set, Tuple

from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- Model loaders -------------------------
def load_llava_model():
    """
    Load full-precision LLaVA and its processor (HF Hub path).
    """
    model_path = "liuhaotian/llava-v1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model, processor


def load_instructblip_model():
    """
    Load full-precision InstructBLIP (HF Hub).
    """
    model_id = "Salesforce/instructblip-vicuna-7b"
    processor = InstructBlipProcessor.from_pretrained(model_id)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model, processor


# --------------------------- I/O helpers --------------------------
def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


@torch.no_grad()
def get_correct_token_probability(model, processor, image, prompt, correct_answer):
    """
    Compute the probability assigned to the correct answer at the final step.

    """
    processed = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**processed)
    logits = outputs.logits  # [B, T, V]
    probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B, V] final token distribution

    # Tokenize the gold answer (no specials)
    token_ids = processor.tokenizer.encode(correct_answer, add_special_tokens=False)
    if len(token_ids) == 0:
        correct_prob = 0.0
        is_correct = False
        predicted_str = ""
    else:
        first_id = token_ids[0]
        correct_prob = float(probs[0, first_id].item())
        pred_id = int(torch.argmax(probs, dim=-1).item())
        pred_tok = processor.tokenizer.convert_ids_to_tokens(pred_id)
        predicted_str = processor.tokenizer.convert_tokens_to_string([pred_tok]).strip()
        is_correct = (pred_id in token_ids)

    return {
        "correct_token_prob": correct_prob,
        "predicted": predicted_str,
        "correct": correct_answer,
        "is_correct": is_correct,
    }


# -------------------- Attention plumbing (hooks) -------------------
def _get_num_heads(module, fallback: int = 32):
    if hasattr(module, "num_heads"):
        return int(module.num_heads)
    if hasattr(module, "num_attention_heads"):
        return int(module.num_attention_heads)
    return fallback


def _find_attn_module_for_layer(model, layer_idx: int):
    """
    Locate LlamaAttention module at '.layers.{i}.self_attn'.
    Works for LLaVA and InstructBLIP (language model part).
    """
    suffix = f".layers.{layer_idx}.self_attn"
    candidates = [(n, m) for n, m in model.named_modules() if n.endswith(suffix)]
    if not candidates:
        # Fallback: looser search if nesting differs
        candidates = [(n, m) for n, m in model.named_modules() if suffix in n]
    if not candidates:
        raise RuntimeError(f"Failed to find self_attn for layer {layer_idx}")
    # Prefer modules under '.language_model.' if both exist
    candidates.sort(key=lambda nm: (0 if ".language_model." in nm[0] else 1, nm[0]))
    return candidates[0][1]


def _make_single_head_o_proj_pre_hook(num_heads: int, head_idx_to_replace: int, mode: str = "mean"):
    """
    Forward-pre-hook for attention.o_proj INPUT ([B, T, H*Dh]).
    Replace one head slice by mean of remaining heads (or zeros).
    """
    h = int(head_idx_to_replace)
    if h < 0 or h >= num_heads:
        raise ValueError(f"head_idx_to_replace out of range: {h} / {num_heads}")

    def pre_hook(module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        x = inputs[0]  # [B, T, D] where D = H * d
        if x.dim() != 3:
            return None
        B, T, D = x.shape
        if D % num_heads != 0:
            return None
        d = D // num_heads

        x_ = x.view(B, T, num_heads, d).clone()
        if mode == "mean":
            if num_heads > 1:
                mask = torch.ones(num_heads, device=x.device, dtype=torch.bool)
                mask[h] = False
                replacement = x_[:, :, mask, :].mean(dim=2)  # [B, T, d]
            else:
                replacement = torch.zeros(B, T, d, dtype=x.dtype, device=x.device)
        elif mode == "zero":
            replacement = torch.zeros(B, T, d, dtype=x.dtype, device=x.device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        x_[:, :, h, :] = replacement
        x_new = x_.view(B, T, D)
        return (x_new,)  # must return a tuple to replace Linear input

    return pre_hook


def register_attn_head_replacement_hook(model, layer_idx: int, head_idx: int):
    """
    Register the pre-hook on the o_proj input of the targeted self-attention.
    """
    attn = _find_attn_module_for_layer(model, layer_idx)
    H = _get_num_heads(attn, fallback=32)
    o_proj = getattr(attn, "o_proj", None)
    if o_proj is None:
        raise RuntimeError(f"Layer {layer_idx} attention has no o_proj; unexpected architecture.")
    hook = _make_single_head_o_proj_pre_hook(H, head_idx, mode="mean")
    handle = o_proj.register_forward_pre_hook(hook, with_kwargs=False)
    return handle


# -------------------------- Main evaluation ------------------------
def evaluate_head_importance(
    model,
    processor,
    data,
    clean_image_dir,
    num_layers=32,
    num_heads=32,
    num_questions=20,
    output_dir="head_importance_results",
    prompt_fn: Optional[Callable[[str], str]] = None,
):
    """
    For each (layer, head), replace that head's contribution BEFORE o_proj mixing.

    NEW:
    - Split samples into baseline-correct vs baseline-incorrect subsets.
    - On the correct subset, aggregate ONLY probability DROPS (ReLU(base - masked)) -> positive heads.
    - On the incorrect subset, aggregate ONLY probability GAINS (ReLU(masked - base)) -> negative heads.

    Outputs:
    - pos_matrix [L, H]: average drop on correct subset (positive heads)
    - neg_matrix [L, H]: average gain on incorrect subset (negative heads)
    - Also saves heatmaps and JSON summaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    selected_questions = data[:num_questions]
    print(f"Evaluating on {len(selected_questions)} questions")

    # Default prompt (LLaVA-style) if not provided
    if prompt_fn is None:
        def prompt_fn(qtext: str) -> str:
            return f"USER: <image>\n{qtext} Please answer with a single word.\nASSISTANT:"

    # -------- Baseline --------
    print("Computing baseline probabilities...")
    baseline_results = []
    missing_images: Set[int] = set()
    for i, q in enumerate(tqdm(selected_questions)):
        image_id = q["imageId"]
        correct_answer = q["answer"]
        prompt = prompt_fn(q["question"])

        try:
            image = load_image(os.path.join(clean_image_dir, f"{image_id}.png"))
        except FileNotFoundError as e:
            print(e)
            missing_images.add(i)
            baseline_results.append({"correct_token_prob": 0.0, "is_correct": False})
            continue

        res = get_correct_token_probability(model, processor, image, prompt, correct_answer)
        baseline_results.append(res)

    baseline_probs = [r["correct_token_prob"] for r in baseline_results]
    baseline_is_correct = [bool(r.get("is_correct", False)) for r in baseline_results]

    # Split indices by baseline correctness (ignore missing images naturally)
    correct_idx = [i for i, ok in enumerate(baseline_is_correct) if ok and i not in missing_images]
    incorrect_idx = [i for i, ok in enumerate(baseline_is_correct) if (not ok) and i not in missing_images]

    baseline_avg_prob = float(np.mean([baseline_probs[i] for i in range(len(baseline_probs)) if i not in missing_images])) if len(baseline_probs) > 0 else 0.0
    baseline_accuracy = float(np.mean([1 if baseline_is_correct[i] else 0 for i in range(len(baseline_is_correct)) if i not in missing_images])) if len(baseline_is_correct) > 0 else 0.0

    print(f"Baseline average probability: {baseline_avg_prob:.4f}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Subset sizes -> correct: {len(correct_idx)}, incorrect: {len(incorrect_idx)}, missing: {len(missing_images)}")

    # -------- Matrices --------
    # pos_matrix: average drop on correct subset (positive heads)
    # neg_matrix: average gain on incorrect subset (negative heads)
    pos_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)
    neg_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)

    # Also keep an overall average (signed) for reference/compatibility
    overall_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)

    all_results: Dict[str, Dict] = {}

    # -------- Grid over heads --------
    for layer_idx in range(num_layers):
        layer_results = {}
        for head_idx in range(num_heads):
            print(f"Testing Layer {layer_idx}, Head {head_idx} (replace with mean of others)...")
            handle = register_attn_head_replacement_hook(model, layer_idx, head_idx)

            # Map question index -> masked stats for alignment with baseline
            masked_info: Dict[int, Dict] = {}

            for i, q in enumerate(tqdm(selected_questions, leave=False)):
                if i in missing_images:
                    continue

                image_id = q["imageId"]
                correct_answer = q["answer"]
                prompt = prompt_fn(q["question"])

                try:
                    image = load_image(os.path.join(clean_image_dir, f"{image_id}.png"))
                except FileNotFoundError:
                    continue

                res = get_correct_token_probability(model, processor, image, prompt, correct_answer)
                base_prob = baseline_results[i]["correct_token_prob"]
                masked_prob = res["correct_token_prob"]
                prob_change = base_prob - masked_prob  # >0 means drop; <0 means improvement

                masked_info[i] = {
                    "masked_prob": masked_prob,
                    "prob_change": prob_change,
                    "masked_is_correct": bool(res.get("is_correct", False)),
                }

            handle.remove()

            # Aggregate on correct subset: only count DROPS (ReLU(base - masked))
            if len(correct_idx) > 0:
                drops = [max(0.0, masked_info[i]["prob_change"]) for i in correct_idx if i in masked_info]
                pos_avg_drop = float(np.mean(drops)) if len(drops) > 0 else 0.0
            else:
                pos_avg_drop = 0.0

            # Aggregate on incorrect subset: only count GAINS (ReLU(masked - base))
            if len(incorrect_idx) > 0:
                gains = [max(0.0, -masked_info[i]["prob_change"]) for i in incorrect_idx if i in masked_info]
                neg_avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
            else:
                neg_avg_gain = 0.0

            # Overall signed average (for reference)
            all_changes = [masked_info[i]["prob_change"] for i in masked_info.keys()]
            overall_avg_change = float(np.mean(all_changes)) if len(all_changes) > 0 else 0.0

            pos_matrix[layer_idx, head_idx] = pos_avg_drop
            neg_matrix[layer_idx, head_idx] = neg_avg_gain
            overall_matrix[layer_idx, head_idx] = overall_avg_change

            # Save per-head summary for JSON (also include counts used)
            layer_results[f"head_{head_idx}"] = {
                "positive_drop_on_correct_subset": pos_avg_drop,
                "negative_gain_on_incorrect_subset": neg_avg_gain,
                "overall_signed_avg_change": overall_avg_change,
                "counts": {
                    "correct_subset_used": int(len([i for i in correct_idx if i in masked_info])),
                    "incorrect_subset_used": int(len([i for i in incorrect_idx if i in masked_info])),
                },
            }

            # ----- Progress plots (optional but handy) -----
            # Positive (correct subset)
            plt.figure(figsize=(12, 10))
            plt.imshow(pos_matrix, cmap="YlOrRd", interpolation="nearest")
            plt.colorbar(label="Positive Score (Avg Drop on Correct Subset)")
            plt.xlabel("Attention Head")
            plt.ylabel("Layer")
            plt.title("Positive Heads Heatmap (Correct Subset Probability Drop)")
            plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
            plt.yticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
            plt.savefig(os.path.join(output_dir, "pos_heads_heatmap_progress.png"))
            plt.close()

            # Negative (incorrect subset)
            plt.figure(figsize=(12, 10))
            plt.imshow(neg_matrix, cmap="YlGnBu", interpolation="nearest")
            plt.colorbar(label="Negative Score (Avg Gain on Incorrect Subset)")
            plt.xlabel("Attention Head")
            plt.ylabel("Layer")
            plt.title("Negative Heads Heatmap (Incorrect Subset Probability Gain)")
            plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
            plt.yticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
            plt.savefig(os.path.join(output_dir, "neg_heads_heatmap_progress.png"))
            plt.close()

            # Persist layer partial results to avoid losing long runs
            with open(os.path.join(output_dir, f"importance_scores_layer{layer_idx}_progress.json"), "w") as f:
                json.dump(layer_results, f, indent=2)

        all_results[f"layer_{layer_idx}"] = layer_results

        # Persist global JSON snapshot after each layer
        with open(os.path.join(output_dir, "head_importance_scores.json"), "w") as f:
            json.dump(
                {
                    "baseline": {
                        "avg_prob": baseline_avg_prob,
                        "accuracy": baseline_accuracy,
                        "num_total": len(selected_questions),
                        "num_missing": len(missing_images),
                        "num_correct_subset": len(correct_idx),
                        "num_incorrect_subset": len(incorrect_idx),
                    },
                    "results": all_results,
                    "matrices": {
                        "positive_correct_subset": pos_matrix.tolist(),
                        "negative_incorrect_subset": neg_matrix.tolist(),
                        "overall_signed_change": overall_matrix.tolist(),
                    },
                },
                f,
                indent=2,
            )

    # -------- Final plots --------
    # Positive heads heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(pos_matrix, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(label="Positive Score (Avg Drop on Correct Subset)")
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.title("Positive Heads Heatmap (Correct Subset Probability Drop)")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.yticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "pos_heads_heatmap_final.png"))
    plt.close()

    # Negative heads heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(neg_matrix, cmap="YlGnBu", interpolation="nearest")
    plt.colorbar(label="Negative Score (Avg Gain on Incorrect Subset)")
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.title("Negative Heads Heatmap (Incorrect Subset Probability Gain)")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.yticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "neg_heads_heatmap_final.png"))
    plt.close()

    # Per-layer averages
    pos_layer_importance = np.mean(pos_matrix, axis=1)  # across heads
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_layers), pos_layer_importance)
    plt.xlabel("Layer")
    plt.ylabel("Avg Positive Score (Drop on Correct)")
    plt.title("Positive Score per Layer")
    plt.xticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "pos_layer_importance.png"))
    plt.close()

    neg_layer_importance = np.mean(neg_matrix, axis=1)  # across heads
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_layers), neg_layer_importance)
    plt.xlabel("Layer")
    plt.ylabel("Avg Negative Score (Gain on Incorrect)")
    plt.title("Negative Score per Layer")
    plt.xticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "neg_layer_importance.png"))
    plt.close()

    # Per-head averages
    pos_head_importance = np.mean(pos_matrix, axis=0)  # across layers
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_heads), pos_head_importance)
    plt.xlabel("Attention Head")
    plt.ylabel("Avg Positive Score (Drop on Correct)")
    plt.title("Positive Score per Attention Head")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.savefig(os.path.join(output_dir, "pos_head_importance.png"))
    plt.close()

    neg_head_importance = np.mean(neg_matrix, axis=0)  # across layers
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_heads), neg_head_importance)
    plt.xlabel("Attention Head")
    plt.ylabel("Avg Negative Score (Gain on Incorrect)")
    plt.title("Negative Score per Attention Head")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.savefig(os.path.join(output_dir, "neg_head_importance.png"))
    plt.close()

    return pos_matrix, neg_matrix, overall_matrix, all_results


def _print_topk_from_matrix(matrix: np.ndarray, k: int, tag: str):
    """
    Utility: print top-k entries (largest values) from a [L, H] matrix.
    """
    flat = matrix.flatten()
    k = min(k, flat.size)
    top_idx = np.argsort(flat)[-k:]
    print(f"\nTop {k} heads by '{tag}':")
    for idx in reversed(top_idx):
        layer_idx = idx // matrix.shape[1]
        head_idx = idx % matrix.shape[1]
        print(f"Layer {layer_idx}, Head {head_idx}: Score = {flat[idx]:.4f}")


# ------------------------------- Main ------------------------------
if __name__ == "__main__":
    # data & dirs
    data_path = "../data/GQA/Object_Level/Animal/Animal.json"
    clean_image_dir = "../data/GQA/Object_Level/Animal/clean"
    out_root = "../data/GQA/Object_Level/Animal/head_importance_results_dual"

    os.makedirs(out_root, exist_ok=True)

    with open(data_path, "r") as f:
        data = json.load(f)

    num_questions = 1000
    num_layers = 32
    num_heads = 32

    # ------------------ LLaVA ------------------
    print("\n[1/2] Loading LLaVA model...")
    llava_model, llava_processor = load_llava_model()
    print("LLaVA loaded.\nStarting evaluation...")
    start = time.time()

    def llava_prompt(qtext: str) -> str:
        return f"USER: <image>\n{qtext} Please answer with a single word.\nASSISTANT:"

    llava_dir = os.path.join(out_root, "llava")
    os.makedirs(llava_dir, exist_ok=True)

    pos_llava, neg_llava, overall_llava, results_llava = evaluate_head_importance(
        llava_model,
        llava_processor,
        data,
        clean_image_dir,
        num_layers=num_layers,
        num_heads=num_heads,
        num_questions=num_questions,
        output_dir=llava_dir,
        prompt_fn=llava_prompt,
    )

    elapsed = (time.time() - start) / 60.0
    print(f"LLaVA evaluation completed in {elapsed:.2f} minutes")

    # Print Top-10 for positive (correct subset drops) and negative (incorrect subset gains)
    _print_topk_from_matrix(pos_llava, k=10, tag="Positive (drop on correct subset)")
    _print_topk_from_matrix(neg_llava, k=10, tag="Negative (gain on incorrect subset)")

    # ---------------- InstructBLIP ----------------
    print("\n[2/2] Loading InstructBLIP model...")
    ib_model, ib_processor = load_instructblip_model()
    print("InstructBLIP loaded.\nStarting evaluation...")
    start = time.time()

    def ib_prompt(qtext: str) -> str:
        return f"Question: {qtext} Answer using a single word."

    ib_dir = os.path.join(out_root, "instructblip")
    os.makedirs(ib_dir, exist_ok=True)

    pos_ib, neg_ib, overall_ib, results_ib = evaluate_head_importance(
        ib_model,
        ib_processor,
        data,
        clean_image_dir,
        num_layers=num_layers,
        num_heads=num_heads,
        num_questions=num_questions,
        output_dir=ib_dir,
        prompt_fn=ib_prompt,
    )

    elapsed = (time.time() - start) / 60.0
    print(f"InstructBLIP evaluation completed in {elapsed:.2f} minutes")

    _print_topk_from_matrix(pos_ib, k=10, tag="Positive (drop on correct subset)")
    _print_topk_from_matrix(neg_ib, k=10, tag="Negative (gain on incorrect subset)")
