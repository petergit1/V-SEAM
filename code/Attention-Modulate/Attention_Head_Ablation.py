#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import random
from typing import List, Tuple, Dict, Optional, Set, Callable

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

# ----------------------------- Config -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Online model identifiers (both models now online)
LLAVA_MODEL_ID = "liuhaotian/llava-v1.5-7b-hf"          # full precision (HF Hub)
INSTRUCTBLIP_MODEL_ID = "Salesforce/instructblip-vicuna-7b"  # full precision (HF Hub)

# Example head sets (two fixed + one random in run_for_model)
EXAMPLE_POSITIVE_HEADS = [(0, 11), (6, 26), (10, 15), (26, 11), (11, 1)]
EXAMPLE_NEGATIVE_HEADS = [(31, 11), (30, 11), (29, 11), (12, 2), (9, 27)]
# third set will be RANDOM_K random heads sampled per model, see run_for_model()

# Data paths
DATA_PATH  = "../data/GQA/Object_Level/Animal/Animal.json"
IMAGE_DIR  = "../data/GQA/Object_Level/Animal/clean"
OUTPUT_DIR = "../data/GQA/Object_Level/Animal/ablation_heads"


# Experiment params
NUM_LAYERS     = 32
NUM_HEADS      = 32
NUM_QUESTIONS  = 1000
RANDOM_K       = 10
RANDOM_TRIALS  = 5
SEED           = 42
# ------------------------------------------------------------------


# ============================ Utilities ============================
def load_llava_model():
    """
    Load full-precision LLaVA model and processor (online from HF Hub).
    """
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
    """
    Load full-precision InstructBLIP model and processor (online from HF Hub).
    """
    processor = InstructBlipProcessor.from_pretrained(INSTRUCTBLIP_MODEL_ID)
    model = InstructBlipForConditionalGeneration.from_pretrained(INSTRUCTBLIP_MODEL_ID)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model, processor


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


@torch.no_grad()
def get_correct_token_probability(model, processor, image, prompt, correct_answer: str) -> Dict:
    """
    Score final-step probability of the first sub-token of the correct answer.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)

    token_ids = processor.tokenizer.encode(correct_answer, add_special_tokens=False)
    if len(token_ids) == 0:
        return {"correct_token_prob": 0.0, "is_correct": False, "predicted": ""}

    correct_prob = float(probs[0, token_ids[0]].item())
    pred_id = int(torch.argmax(probs, dim=-1).item())
    pred_tok = processor.tokenizer.convert_ids_to_tokens(pred_id)
    predicted = processor.tokenizer.convert_tokens_to_string([pred_tok]).strip()
    return {"correct_token_prob": correct_prob, "is_correct": pred_id in token_ids, "predicted": predicted}


def _get_num_heads_from_attn(module, fallback: int = 32) -> int:
    if hasattr(module, "num_heads"):
        return int(module.num_heads)
    if hasattr(module, "num_attention_heads"):
        return int(module.num_attention_heads)
    return fallback


def _iter_named_modules(model):
    for name, mod in model.named_modules():
        yield name, mod


def _find_attn_module_for_layer(model, layer_idx: int):
    """
    Find LlamaAttention module for a given layer by suffix match.
    Works for both LLaVA and InstructBLIP (LLaMA-based).
    """
    suffix = f".layers.{layer_idx}.self_attn"
    candidates = []
    for name, mod in _iter_named_modules(model):
        if name.endswith(suffix):
            candidates.append((name, mod))
    if not candidates:
        for name, mod in _iter_named_modules(model):
            if suffix in name and getattr(mod, "forward", None) is not None:
                candidates.append((name, mod))
    if not candidates:
        raise RuntimeError(f"Could not find self_attn for layer {layer_idx}")
    candidates.sort(key=lambda nm: (0 if ".language_model." in nm[0] else 1, nm[0]))
    return candidates[0][1]


def _group_heads_by_layer(heads: List[Tuple[int,int]]) -> Dict[int, Set[int]]:
    grouped: Dict[int, Set[int]] = {}
    for l, h in heads:
        grouped.setdefault(l, set()).add(h)
    return grouped


def _make_o_proj_pre_hook(num_heads: int, heads_to_replace: List[int], mode: str = "zero"):
    """
    Pre-hook on attention.o_proj: replace specified heads BEFORE mixing by o_proj.
    Input to Linear is [B, T, H*D_head]. We reshape to [B, T, H, d] and
    replace selected head slices by zeros (mask=0).
    """
    heads_to_replace = sorted(set(int(h) for h in heads_to_replace if 0 <= int(h) < num_heads))

    def pre_hook(module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        x = inputs[0]  # [B, T, D=H*d]
        if x.dim() != 3:
            return None
        B, T, D = x.shape
        if D % num_heads != 0:
            return None
        d = D // num_heads

        x_ = x.view(B, T, num_heads, d).clone()

        zero_chunk = torch.zeros(B, T, d, dtype=x.dtype, device=x.device)
        for h in heads_to_replace:
            x_[:, :, h, :] = zero_chunk

        x_new = x_.view(B, T, D)
        return (x_new,)  # replace first arg to o_proj
    return pre_hook


def register_heads_hooks(model, heads: List[Tuple[int, int]]) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register pre-hooks on attention.o_proj inputs for the given (layer, head) pairs.
    Group per layer so each layer has one hook. (zero-masking)
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    layer_to_heads = _group_heads_by_layer(heads)

    for layer_idx, head_set in layer_to_heads.items():
        attn = _find_attn_module_for_layer(model, layer_idx)
        H = _get_num_heads_from_attn(attn, fallback=NUM_HEADS)
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None:
            raise RuntimeError(f"Layer {layer_idx} attention has no o_proj; unexpected architecture.")
        hook = _make_o_proj_pre_hook(H, sorted(head_set), mode="zero")
        handle = o_proj.register_forward_pre_hook(hook, with_kwargs=False)
        handles.append(handle)
    return handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def pick_random_heads(k: int, num_layers: int = NUM_LAYERS, num_heads: int = NUM_HEADS, *, rng: Optional[random.Random] = None) -> List[Tuple[int, int]]:
    all_pairs = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    (rng or random).shuffle(all_pairs)
    return all_pairs[:max(0, min(k, len(all_pairs)))]


def load_questions(path: str, limit: Optional[int] = None) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return items[:limit] if limit is not None else items
# ==================================================================


# =========================== Experiments ===========================
def compute_baseline(
    model, processor, questions: List[dict], image_dir: str, prompt_fn: Callable[[str], str]
) -> Dict[str, List[Dict]]:
    """
    Compute baseline probability and correctness for each question (model-specific prompt_fn).
    """
    results = []
    for q in tqdm(questions, desc="Baseline"):
        img_path = os.path.join(image_dir, f"{q['imageId']}.jpg")
        try:
            image = load_image(img_path)
        except FileNotFoundError:
            results.append({"correct_token_prob": 0.0, "is_correct": False})
            continue

        prompt = prompt_fn(q["question"])
        res = get_correct_token_probability(model, processor, image, prompt, q["answer"])
        results.append(res)
    return {
        "per_question": results,
        "avg_prob": float(np.mean([r["correct_token_prob"] for r in results] or [0.0])),
        "acc": float(np.mean([1 if r.get("is_correct", False) else 0 for r in results] or [0.0])),
    }


def evaluate_condition(
    model, processor, questions: List[dict], image_dir: str, heads: List[Tuple[int, int]],
    label: str, out_dir: str, baseline_per_q: List[Dict], prompt_fn: Callable[[str], str]
) -> Dict:
    """
    Evaluate one ablation condition (selected heads zero-masked before o_proj).
    """
    handles = register_heads_hooks(model, heads)
    per_q = []

    for i, q in enumerate(tqdm(questions, desc=label)):
        img_path = os.path.join(image_dir, f"{q['imageId']}.jpg")
        try:
            image = load_image(img_path)
        except FileNotFoundError:
            per_q.append({"correct_token_prob": 0.0, "is_correct": False, "prob_drop": 0.0})
            continue

        prompt = prompt_fn(q["question"])
        res = get_correct_token_probability(model, processor, image, prompt, q["answer"])
        base_prob = baseline_per_q[i]["correct_token_prob"]
        res["prob_drop"] = base_prob - res["correct_token_prob"]
        per_q.append(res)

    remove_hooks(handles)

    avg_prob = float(np.mean([r["correct_token_prob"] for r in per_q] or [0.0]))
    acc = float(np.mean([1 if r.get("is_correct", False) else 0 for r in per_q] or [0.0]))
    avg_drop = float(np.mean([r["prob_drop"] for r in per_q] or [0.0]))

    payload = {
        "label": label,
        "heads": heads,
        "avg_prob": avg_prob,
        "acc": acc,
        "avg_prob_drop": avg_drop,
        "per_question": per_q,
    }

    with open(os.path.join(out_dir, f"{label.replace(' ', '_')}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def evaluate_random_trials(
    model, processor, questions: List[dict], image_dir: str, k: int, trials: int,
    out_dir: str, baseline_per_q: List[Dict], seed: int, prompt_fn: Callable[[str], str]
) -> Dict:
    """
    Run multiple random ablation trials and summarize statistics (zero-masked heads).
    """
    rng = random.Random(seed)
    trial_summaries = []
    per_trial = []

    for t in range(trials):
        rng.seed(seed + t)
        heads = pick_random_heads(k, rng=rng)
        result = evaluate_condition(
            model, processor, questions, image_dir, heads, f"random_trial_{t+1}", out_dir, baseline_per_q, prompt_fn
        )
        per_trial.append(result)
        trial_summaries.append((result["avg_prob_drop"], result["acc"]))

    avg_drop = float(np.mean([d for d, _ in trial_summaries] or [0.0]))
    std_drop = float(np.std([d for d, _ in trial_summaries] or [0.0]))
    avg_acc = float(np.mean([a for _, a in trial_summaries] or [0.0]))
    std_acc = float(np.std([a for _, a in trial_summaries] or [0.0]))

    summary = {
        "k": k,
        "trials": trials,
        "avg_prob_drop": avg_drop,
        "std_prob_drop": std_drop,
        "avg_acc": avg_acc,
        "std_acc": std_acc,
    }

    with open(os.path.join(out_dir, "random_trials_summary.json"), "w") as f:
        json.dump({"summary": summary, "per_trial": per_trial}, f, indent=2)

    return summary
# ==================================================================


# ============================== Plots ==============================
def plot_bar(values: Dict[str, float], ylabel: str, title: str, out_path: str):
    names = list(values.keys())
    nums = [values[k] for k in names]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(names)), nums)
    plt.xticks(np.arange(len(names)), names, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
# ==================================================================


# ============================ Run per model =========================
def run_for_model(model_name: str, model, processor, questions: List[dict], out_root: str):
    """
    Reuse the same evaluation logic for a specific model, with its own prompt style and subdir.
    Third condition uses RANDOM_K random heads (sampled once per model with a deterministic seed).
    """
    out_dir = os.path.join(out_root, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Prompts
    if model_name == "llava":
        def prompt_fn(qtext: str) -> str:
            return (
                "USER: <image>\n"
                f"{qtext}\n"
                "Please answer based on the object at the starred position in the image with a single word.\n"
                "ASSISTANT:"
            )
        model_seed_offset = 0
    else:  # instructblip
        def prompt_fn(qtext: str) -> str:
            return f"Question: {qtext} Answer using a single word."
        model_seed_offset = 997  # different offset to keep per-model sampling stable but distinct

    # Baseline
    baseline = compute_baseline(model, processor, questions, IMAGE_DIR, prompt_fn)
    with open(os.path.join(out_dir, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)

    # Random set (sample once, deterministic)
    rng = random.Random(SEED + model_seed_offset)
    random_heads_once = pick_random_heads(RANDOM_K, rng=rng)
    with open(os.path.join(out_dir, "random_heads_once.json"), "w") as f:
        json.dump({"heads": random_heads_once}, f, indent=2)

    # Three conditions: positive / negative / random (once)
    conditions = {
        "positive_set": EXAMPLE_POSITIVE_HEADS,
        "negative_set": EXAMPLE_NEGATIVE_HEADS,
        "random_set_once": random_heads_once,
    }

    summaries = {}
    for label, heads in conditions.items():
        print(f"[{model_name}] Evaluating: {label} ({len(heads)} heads)")
        res = evaluate_condition(
            model, processor, questions, IMAGE_DIR, heads, label, out_dir, baseline["per_question"], prompt_fn
        )
        summaries[f"{label}_avg_prob_drop"] = res["avg_prob_drop"]
        summaries[f"{label}_acc"] = res["acc"]

    # Random trials (multi-run)
    print(f"[{model_name}] Evaluating random ablations: k={RANDOM_K}, trials={RANDOM_TRIALS}")
    rnd_summary = evaluate_random_trials(
        model, processor, questions, IMAGE_DIR, RANDOM_K, RANDOM_TRIALS, out_dir, baseline["per_question"], SEED + model_seed_offset, prompt_fn
    )
    summaries["random_trials_avg_prob_drop"] = rnd_summary["avg_prob_drop"]
    summaries["random_trials_avg_prob_drop_std"] = rnd_summary["std_prob_drop"]
    summaries["random_trials_acc"] = rnd_summary["avg_acc"]
    summaries["random_trials_acc_std"] = rnd_summary["std_acc"]

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    # Plots
    plot_bar(
        {
            "positive": summaries.get("positive_set_example_avg_prob_drop", summaries.get("positive_set_avg_prob_drop", 0.0)),
            "negative": summaries.get("negative_set_example_avg_prob_drop", summaries.get("negative_set_avg_prob_drop", 0.0)),
            "random_once": summaries.get("random_set_once_avg_prob_drop", 0.0),
            "random_trials": summaries.get("random_trials_avg_prob_drop", 0.0),
        },
        ylabel="Avg Probability Drop (baseline - ablated)",
        title=f"Head Ablation: Probability Drop ({model_name})",
        out_path=os.path.join(out_dir, "ablation_prob_drop.png"),
    )

    plot_bar(
        {
            "positive": summaries.get("positive_set_example_acc", summaries.get("positive_set_acc", 0.0)),
            "negative": summaries.get("negative_set_example_acc", summaries.get("negative_set_acc", 0.0)),
            "random_once": summaries.get("random_set_once_acc", 0.0),
            "random_trials": summaries.get("random_trials_acc", 0.0),
        },
        ylabel="Accuracy",
        title=f"Head Ablation: Accuracy ({model_name})",
        out_path=os.path.join(out_dir, "ablation_accuracy.png"),
    )

    print(f"[{model_name}] Done. Results saved to: {out_dir}")


# =============================== Main ==============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data[:NUM_QUESTIONS]

    print("Loading LLaVA (online)...")
    llava_model, llava_processor = load_llava_model()
    print("LLaVA ready.")
    run_for_model("llava", llava_model, llava_processor, questions, OUTPUT_DIR)

    print("Loading InstructBLIP (online)...")
    ib_model, ib_processor = load_instructblip_model()
    print("InstructBLIP ready.")
    run_for_model("instructblip", ib_model, ib_processor, questions, OUTPUT_DIR)

    print("All done.")


if __name__ == "__main__":
    main()


