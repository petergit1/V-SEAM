#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

# ============================== Config ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Online model identifiers (full precision)
LLAVA_MODEL_ID = "liuhaotian/llava-v1.5-7b-hf"
INSTRUCTBLIP_MODEL_ID = "Salesforce/instructblip-vicuna-7b"

# Data & output
DATA_PATH       = "../data/GQA/Object_Level/Animal/Animal.json"
CLEAN_IMAGE_DIR = "../data/GQA/Object_Level/Animal/clean"
OUTPUT_DIR      = "../data/GQA/Object_Level/Animal/head_importance_dual_models"


# Eval ranges
NUM_LAYERS = 32
NUM_HEADS  = 32
NUM_QUESTIONS_IMPORTANCE = 1000    # used in importance evaluation
NUM_QUESTIONS_EDIT_EVAL  = 200     # quick eval after edits

# Edit config
EDIT_LAMBDA = 1.0        # scale = 1 + EDIT_LAMBDA * signed_score
EPS_FOR_EDIT = 0.0       # |score| below this will be ignored
CLIP_RANGE = (0.0, 2.0)  # clip scale for stability (set None to disable)
# ===================================================================


# ============================== I/O ==============================
def load_llava_model():
    """Load full-precision LLaVA and its processor (online)."""
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model, processor


def load_instructblip_model():
    """Load full-precision InstructBLIP and its processor (online)."""
    processor = InstructBlipProcessor.from_pretrained(INSTRUCTBLIP_MODEL_ID)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        INSTRUCTBLIP_MODEL_ID
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model, processor


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


@torch.no_grad()
def get_correct_token_probability(model, processor, image, prompt, correct_answer) -> Dict:
    """Compute final-step probability assigned to the correct answer's first sub-token."""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)

    token_ids = processor.tokenizer.encode(correct_answer, add_special_tokens=False)
    if len(token_ids) == 0:
        return {"correct_token_prob": 0.0, "predicted": "", "correct": correct_answer, "is_correct": False}

    correct_prob = float(probs[0, token_ids[0]].item())
    pred_id = int(torch.argmax(probs, dim=-1).item())
    pred_tok = processor.tokenizer.convert_ids_to_tokens(pred_id)
    pred_str = processor.tokenizer.convert_tokens_to_string([pred_tok]).strip()
    is_correct = pred_id in token_ids
    return {
        "correct_token_prob": correct_prob,
        "predicted": pred_str,
        "correct": correct_answer,
        "is_correct": is_correct,
    }
# ===================================================================


# =========================== Hook Utilities =========================
def _get_num_heads(attn_module, fallback: int = 32) -> int:
    if hasattr(attn_module, "num_heads"):
        return int(attn_module.num_heads)
    if hasattr(attn_module, "num_attention_heads"):
        return int(attn_module.num_attention_heads)
    return fallback


def _find_attn_module_for_layer(model, layer_idx: int):
    """
    Robustly locate the LlamaAttention of a given layer.
    Prefer names that end with '.layers.{i}.self_attn' and under '.language_model.' if possible.
    """
    suffix = f".layers.{layer_idx}.self_attn"
    candidates = [(n, m) for n, m in model.named_modules() if n.endswith(suffix)]
    if not candidates:
        candidates = [(n, m) for n, m in model.named_modules() if suffix in n]
    if not candidates:
        raise RuntimeError(f"Failed to find self_attn for layer {layer_idx}")
    candidates.sort(key=lambda nm: (0 if ".language_model." in nm[0] else 1, nm[0]))
    return candidates[0][1]


def _make_single_head_o_proj_pre_hook(num_heads: int, head_idx_to_replace: int, mode: str = "mean"):
    """
    Create a forward_pre_hook for attention.o_proj INPUT (shape [B, T, H*Dh]).
    Replace one head chunk by the mean of remaining heads (or zeros if mode='zero').
    """
    h = int(head_idx_to_replace)
    if h < 0 or h >= num_heads:
        raise ValueError(f"head_idx_to_replace out of range: {h} / {num_heads}")

    def pre_hook(module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        x = inputs[0]  # [B, T, D], D = H * d
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
        return (x_new,)

    return pre_hook


def register_attn_head_replacement_hook(model, layer_idx: int, head_idx: int, mode: str = "mean"):
    """
    Register a forward_pre_hook on attention.o_proj INPUT for (layer, head).
    """
    attn = _find_attn_module_for_layer(model, layer_idx)
    H = _get_num_heads(attn, fallback=32)
    o_proj = getattr(attn, "o_proj", None)
    if o_proj is None:
        raise RuntimeError(f"Layer {layer_idx} attention has no o_proj; unexpected architecture.")
    hook = _make_single_head_o_proj_pre_hook(H, head_idx, mode=mode)
    handle = o_proj.register_forward_pre_hook(hook, with_kwargs=False)
    return handle


def _make_layer_scaling_o_proj_pre_hook(num_heads: int, scales_1d: np.ndarray, clamp: Optional[Tuple[float,float]] = (0.0, 2.0)):
    """
    Per-layer head scaling hook: x[:, :, h, :] *= scale[h] before o_proj mixing.
    """
    assert scales_1d.shape == (num_heads,)
    scales_np = scales_1d.copy()

    def pre_hook(module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        x = inputs[0]  # [B, T, D] where D=H*Dh
        if x.dim() != 3:
            return None
        B, T, D = x.shape
        if D % num_heads != 0:
            return None
        d = D // num_heads

        s = np.clip(scales_np, clamp[0], clamp[1]) if clamp is not None else scales_np
        scales = torch.tensor(s, dtype=x.dtype, device=x.device)  # [H]

        x_ = x.view(B, T, num_heads, d)
        x_ = x_ * scales.view(1, 1, num_heads, 1)
        x_new = x_.view(B, T, D)
        return (x_new,)

    return pre_hook


def register_head_edit_hooks(model, signed_score_matrix: np.ndarray, edit_lambda: float = 1.0,
                             epsilon: float = 0.0, clip: Optional[Tuple[float,float]] = (0.0, 2.0)):
    """
    Install head-scaling hooks based on signed scores:
      scale[h] = 1 + edit_lambda * score[h], with optional clipping.
    Heads with |score| < epsilon are treated as zero.
    """
    L, H = signed_score_matrix.shape
    handles = []
    for l in range(L):
        s = signed_score_matrix[l].copy()
        s[np.abs(s) < epsilon] = 0.0
        scales = 1.0 + edit_lambda * s
        attn = _find_attn_module_for_layer(model, l)
        H_real = _get_num_heads(attn, fallback=H)
        assert H_real == H, f"Head count mismatch at layer {l}: {H_real} vs {H}"
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None:
            raise RuntimeError(f"Layer {l} attention has no o_proj; unexpected architecture.")
        hook = _make_layer_scaling_o_proj_pre_hook(H, scales, clamp=clip)
        handle = o_proj.register_forward_pre_hook(hook, with_kwargs=False)
        handles.append(handle)
    return handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
# ===================================================================


# ======================= Importance Evaluation ======================
def compute_top_heads(matrix: np.ndarray, k: int = 10):
    flat = matrix.flatten()
    idxs = np.argsort(flat)[-k:]
    top = []
    L, H = matrix.shape
    for i in reversed(idxs):
        layer = i // H
        head = i % H
        top.append((int(layer), int(head), float(flat[i])))
    return top


def evaluate_head_importance(
    model,
    processor,
    data,
    clean_image_dir,
    prompt_fn: Callable[[str], str],
    num_layers=32,
    num_heads=32,
    num_questions=20,
    output_dir="head_importance_results",
    epsilon=0.0,
    replace_mode: str = "mean",  # mean | zero
):
    """
    For each (layer, head), replace that head's contribution BEFORE o_proj mixing
    and measure base_prob - patched_prob as the drop score.
    """
    os.makedirs(output_dir, exist_ok=True)

    selected = data[:num_questions]
    print(f"Evaluating on {len(selected)} questions")

    # Baseline
    print("Computing baseline probabilities...")
    baseline = []
    for q in tqdm(selected):
        image_id = q["imageId"]
        correct_answer = q["answer"]
        question_text = q["question"]
        prompt = prompt_fn(question_text)
        try:
            image = load_image(os.path.join(clean_image_dir, f"{image_id}.png"))
        except FileNotFoundError as e:
            print(e)
            baseline.append({"correct_token_prob": 0.0, "is_correct": False})
            continue
        res = get_correct_token_probability(model, processor, image, prompt, correct_answer)
        baseline.append(res)

    base_probs = [r["correct_token_prob"] for r in baseline]
    base_avg_prob = float(np.mean(base_probs))
    base_acc = float(np.mean([1 if r.get("is_correct", False) else 0 for r in baseline]))
    print(f"Baseline average probability: {base_avg_prob:.4f}")
    print(f"Baseline accuracy: {base_acc:.4f}")

    prob_drop_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)
    positive_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)
    negative_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)
    all_results = {}

    for layer_idx in range(num_layers):
        layer_results = {}
        for head_idx in range(num_heads):
            print(f"Testing Layer {layer_idx}, Head {head_idx} (replace mode: {replace_mode})...")
            handle = register_attn_head_replacement_hook(model, layer_idx, head_idx, mode=replace_mode)

            masked = []
            for i, q in enumerate(tqdm(selected)):
                image_id = q["imageId"]
                correct_answer = q["answer"]
                question_text = q["question"]
                prompt = prompt_fn(question_text)

                try:
                    image = load_image(os.path.join(clean_image_dir, f"{image_id}.png"))
                except FileNotFoundError:
                    continue

                res = get_correct_token_probability(model, processor, image, prompt, correct_answer)
                base_p = baseline[i]["correct_token_prob"]
                res["prob_drop"] = base_p - res["correct_token_prob"]
                masked.append(res)

            handle.remove()

            drops = [r["prob_drop"] for r in masked] if masked else [0.0]
            avg_drop = float(np.mean(drops))
            prob_drop_matrix[layer_idx, head_idx] = avg_drop
            if avg_drop > epsilon:
                positive_matrix[layer_idx, head_idx] = avg_drop
            elif avg_drop < -epsilon:
                negative_matrix[layer_idx, head_idx] = float(-avg_drop)

            layer_results[f"head_{head_idx}"] = {
                "avg_prob_drop": avg_drop,
                "classification": "positive" if avg_drop > epsilon else ("negative" if avg_drop < -epsilon else "neutral"),
            }

            # Progressive heatmap
            plt.figure(figsize=(12, 10))
            plt.imshow(prob_drop_matrix, cmap="PiYG", interpolation="nearest")
            plt.colorbar(label="Prob Drop (base - patched); >0 positive head, <0 negative head")
            plt.xlabel("Attention Head")
            plt.ylabel("Layer")
            plt.title("Attention Head Effect (Positive vs Negative)")
            plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
            plt.yticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
            plt.savefig(os.path.join(output_dir, "posneg_heatmap_progress.png"))
            plt.close()

        all_results[f"layer_{layer_idx}"] = layer_results

        with open(os.path.join(output_dir, "posneg_scores_progress.json"), "w") as f:
            json.dump(
                {
                    "baseline_avg_prob": base_avg_prob,
                    "baseline_accuracy": base_acc,
                    "prob_drop_matrix": prob_drop_matrix.tolist(),
                    "positive_matrix": positive_matrix.tolist(),
                    "negative_matrix": negative_matrix.tolist(),
                    "results": all_results,
                },
                f,
                indent=2,
            )

    # Final heatmaps
    plt.figure(figsize=(12, 10))
    plt.imshow(prob_drop_matrix, cmap="PiYG", interpolation="nearest")
    plt.colorbar(label="Prob Drop (base - patched); >0 positive head, <0 negative head")
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.title("Attention Head Effect (Positive vs Negative)")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.yticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "posneg_heatmap_final.png"))
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.imshow(positive_matrix, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(label="Positive Head Score (Prob Drop > 0)")
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.title("Positive Heads (Helpful)")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.yticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "positive_heads_heatmap.png"))
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.imshow(negative_matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Negative Head Score (|Prob Drop| where Prob Drop < 0)")
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")
    plt.title("Negative Heads (Harmful)")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.yticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "negative_heads_heatmap.png"))
    plt.close()

    # Per-layer averages
    pos_layer_avg = np.mean(positive_matrix, axis=1)
    neg_layer_avg = np.mean(negative_matrix, axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_layers), pos_layer_avg)
    plt.xlabel("Layer"); plt.ylabel("Avg Positive Score"); plt.title("Average Positive Head Score per Layer")
    plt.xticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "layer_avg_positive.png")); plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_layers), neg_layer_avg)
    plt.xlabel("Layer"); plt.ylabel("Avg Negative Score"); plt.title("Average Negative Head Score per Layer")
    plt.xticks(np.arange(num_layers), [str(i) for i in range(num_layers)])
    plt.savefig(os.path.join(output_dir, "layer_avg_negative.png")); plt.close()

    # Per-head averages
    pos_head_avg = np.mean(positive_matrix, axis=0)
    neg_head_avg = np.mean(negative_matrix, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_heads), pos_head_avg)
    plt.xlabel("Attention Head"); plt.ylabel("Avg Positive Score"); plt.title("Average Positive Score per Head")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.savefig(os.path.join(output_dir, "head_avg_positive.png")); plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_heads), neg_head_avg)
    plt.xlabel("Attention Head"); plt.ylabel("Avg Negative Score"); plt.title("Average Negative Score per Head")
    plt.xticks(np.arange(num_heads), [str(i) for i in range(num_heads)])
    plt.savefig(os.path.join(output_dir, "head_avg_negative.png")); plt.close()

    top_positive = compute_top_heads(positive_matrix, k=20)
    top_negative = compute_top_heads(negative_matrix, k=20)

    with open(os.path.join(output_dir, "head_posneg_summary.json"), "w") as f:
        json.dump(
            {
                "baseline_avg_prob": base_avg_prob,
                "baseline_accuracy": base_acc,
                "top_positive": top_positive,
                "top_negative": top_negative,
                "prob_drop_matrix": prob_drop_matrix.tolist(),
                "positive_matrix": positive_matrix.tolist(),
                "negative_matrix": negative_matrix.tolist(),
            },
            f,
            indent=2,
        )

    return prob_drop_matrix, positive_matrix, negative_matrix, {
        "baseline_avg_prob": base_avg_prob, "baseline_acc": base_acc,
        "top_positive": top_positive, "top_negative": top_negative
    }
# ===================================================================


# ============================ Edit Evaluation =======================
def evaluate_with_edits(model, processor, data, clean_image_dir, prompt_fn: Callable[[str], str], num_questions=200):
    selected = data[:num_questions]
    probs, accs = [], []
    for q in tqdm(selected, desc="Eval with edits"):
        image_id = q["imageId"]
        correct_answer = q["answer"]
        question_text = q["question"]
        prompt = prompt_fn(question_text)
        try:
            image = load_image(os.path.join(clean_image_dir, f"{image_id}.png"))
        except FileNotFoundError:
            continue
        res = get_correct_token_probability(model, processor, image, prompt, correct_answer)
        probs.append(res["correct_token_prob"])
        accs.append(1 if res["is_correct"] else 0)
    return float(np.mean(probs or [0.0])), float(np.mean(accs or [0.0]))
# ===================================================================


# =============================== Runner =============================
def run_for_model(model_key: str, model, processor, data: List[dict], out_root: str):
    """
    Run importance + editing for a specific model with its own prompt style and output subdir.
    """
    subdir = os.path.join(out_root, model_key)
    os.makedirs(subdir, exist_ok=True)

    # Model-specific prompt builders
    if model_key == "llava":
        def prompt_fn(qtext: str) -> str:
            return f"USER: <image>\n{qtext} Please answer with a single word.\nASSISTANT:"
        replace_mode = "mean"  # keep same as your original script for importance
    else:  # instructblip
        def prompt_fn(qtext: str) -> str:
            return f"Question: {qtext} Answer using a single word."
        replace_mode = "mean"

    # 1) Importance
    print(f"[{model_key}] Starting importance evaluation on {NUM_QUESTIONS_IMPORTANCE} questions...")
    start = time.time()
    prob_drop_matrix, pos_mat, neg_mat, meta = evaluate_head_importance(
        model,
        processor,
        data,
        CLEAN_IMAGE_DIR,
        prompt_fn=prompt_fn,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_questions=NUM_QUESTIONS_IMPORTANCE,
        output_dir=subdir,
        epsilon=0.0,
        replace_mode=replace_mode,
    )
    elapsed = (time.time() - start) / 60.0
    print(f"[{model_key}] Importance evaluation completed in {elapsed:.2f} minutes")

    # 2) Register edit hooks based on signed scores (prob_drop_matrix)
    print(f"[{model_key}] Registering head-edit hooks...")
    edit_handles = register_head_edit_hooks(
        model,
        signed_score_matrix=prob_drop_matrix,
        edit_lambda=EDIT_LAMBDA,
        epsilon=EPS_FOR_EDIT,
        clip=CLIP_RANGE,
    )

    # 3) Quick eval with edits
    print(f"[{model_key}] Evaluating with edits on {NUM_QUESTIONS_EDIT_EVAL} questions...")
    avg_prob_edit, acc_edit = evaluate_with_edits(
        model, processor, data, CLEAN_IMAGE_DIR, prompt_fn=prompt_fn, num_questions=NUM_QUESTIONS_EDIT_EVAL
    )
    print(f"[{model_key}] [With edits] Avg prob: {avg_prob_edit:.4f}, Acc: {acc_edit:.4f}")

    # Save matrices and edit config
    np.save(os.path.join(subdir, "prob_drop_matrix.npy"), prob_drop_matrix)
    with open(os.path.join(subdir, "edit_config.json"), "w") as f:
        json.dump(
            {
                "edit_lambda": EDIT_LAMBDA,
                "epsilon_for_edit": EPS_FOR_EDIT,
                "clip_range": CLIP_RANGE,
                "with_edits_avg_prob": avg_prob_edit,
                "with_edits_acc": acc_edit,
                "baseline_avg_prob": meta["baseline_avg_prob"],
                "baseline_acc": meta["baseline_acc"],
            },
            f,
            indent=2,
        )

    remove_hooks(edit_handles)
# ===================================================================


# ================================ Main =============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # ---- LLaVA ----
    print("Loading LLaVA (online full precision)...")
    llava_model, llava_processor = load_llava_model()
    print("LLaVA loaded.")
    run_for_model("llava", llava_model, llava_processor, data, OUTPUT_DIR)

    # ---- InstructBLIP ----
    print("Loading InstructBLIP (online full precision)...")
    ib_model, ib_processor = load_instructblip_model()
    print("InstructBLIP loaded.")
    run_for_model("instructblip", ib_model, ib_processor, data, OUTPUT_DIR)

    print("All done. Results saved under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


