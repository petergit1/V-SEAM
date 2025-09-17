#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
from typing import List, Dict, Optional
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

# ====================== Config ======================
MODEL_TYPE = "llava"  # "llava" | "instructblip"

# HuggingFace model IDs (online download)
LLAVA_MODEL_ID = "liuhaotian/llava-v1.5-7b"
INSTRUCTBLIP_MODEL_ID = "Salesforce/instructblip-vicuna-7b"

# Number of leading tokens in the LM input corresponding to the image prefix.
LLAVA_IMAGE_TOKENS = 576

INSTRUCTBLIP_FALLBACK_QTOK = 32

NUM_QUESTIONS = 1000
# ====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


@torch.no_grad()
def forward_logits(model, processed_inputs) -> torch.Tensor:
    """Single forward pass returning logits of shape [B, T, V]."""
    outputs = model(**processed_inputs)
    return outputs.logits


def unwrap_output(output):
    """Normalize module outputs to a Tensor (pull hidden_states when tuple)."""
    if isinstance(output, tuple):
        return output[0]
    return output


def get_target_token_ids(tokenizer, answer: str) -> List[int]:
    """
    Robust tokenization to ids without special tokens.
    Falls back to lowercased form if the original fails.
    """
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer.strip().lower(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer '{answer}' cannot be tokenized to any ids.")
    return ids


def last_token_scores(logits: torch.Tensor, token_ids: List[int]) -> Dict[str, float]:
    """
    Return both probability and logit metrics at the last position:
      - prob : sum of P(vocab_id ∈ token_ids)
      - logit: mean of logits over token_ids
    """
    last = logits[:, -1, :]  # [B, V]
    probs = torch.softmax(last, dim=-1)
    prob_val = float(sum(probs[0, t].item() for t in token_ids))
    logit_val = float(torch.stack([last[0, t] for t in token_ids]).mean().item())
    return {"prob": prob_val, "logit": logit_val}


# ----------------- Model Adapters -----------------
class BaseAdapter:
    """Abstract adapter for loading model/processor and providing model-specific logic."""
    name = "base"

    def load(self):
        raise NotImplementedError

    def preprocess(self, image: Image.Image, text: str):
        raise NotImplementedError

    def prompt(self, question: str) -> str:
        raise NotImplementedError

    def tokenizer(self):
        return self.processor.tokenizer

    def image_token_count(self) -> int:
        raise NotImplementedError

    def default_layers(self) -> List[str]:
        """Default to whole Transformer layers (you can override)."""
        return [f"language_model.model.layers.{i}" for i in range(32)]


class LLaVAAdapter(BaseAdapter):
    name = "llava"

    def __init__(self, model_id: str, image_tokens: int):
        self.model_id = model_id
        self._image_tokens = image_tokens

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        )
        self.model.to(device).eval()
        self.model.config.use_cache = False
        return self.model, self.processor

    def preprocess(self, image: Image.Image, text: str):
        return self.processor(images=image, text=text, return_tensors="pt").to(device)

    def prompt(self, question: str) -> str:
        return f"USER: <image>\n{question} Please answer with a single word.\nASSISTANT:"

    def image_token_count(self) -> int:
        return self._image_tokens


class InstructBLIPAdapter(BaseAdapter):
    name = "instructblip"

    def __init__(self, model_id: str, use_4bit: bool = True):
        self.model_id = model_id
        self.use_4bit = use_4bit

    def load(self):
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        quant_kwargs = {}
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            )
        self.processor = InstructBlipProcessor.from_pretrained(self.model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_id, **quant_kwargs
        )
        # 4-bit models typically should not be moved with .to(device)
        try:
            if not self.use_4bit:
                self.model.to(device)
        except Exception:
            pass
        self.model.eval()
        return self.model, self.processor

    def preprocess(self, image: Image.Image, text: str):
        return self.processor(images=image, text=text, return_tensors="pt").to(device)

    def prompt(self, question: str) -> str:
        return f"Question: {question} Answer using a single word."

    def image_token_count(self) -> int:
        """
        Use Q-Former query token count if available; otherwise fall back.
        """
        qconf = getattr(getattr(self.model, "config", None), "qformer_config", None)
        if qconf and hasattr(qconf, "num_query_tokens"):
            return int(qconf.num_query_tokens)
        return INSTRUCTBLIP_FALLBACK_QTOK


# ----------------- Activation caching & patching -----------------
def cache_clean_activations(
    model, layers_to_patch: List[str], processed_clean
) -> Dict[str, torch.Tensor]:
    """
    Run a clean forward pass and cache the activations at specified layers.
    """
    clean_acts = {}
    hooks = []

    def make_hook(layer_name: str):
        def _hook(module, inputs, output):
            out = unwrap_output(output)
            if layer_name not in clean_acts:
                clean_acts[layer_name] = out.detach().clone()
        return _hook

    # Register hooks, run one forward, then remove hooks.
    for layer_name in layers_to_patch:
        for name, module in model.named_modules():
            if name == layer_name:
                hooks.append(module.register_forward_hook(make_hook(layer_name)))
                break

    _ = forward_logits(model, processed_clean)

    for h in hooks:
        h.remove()
    return clean_acts


def make_group_patch_hook(clean_act: torch.Tensor, img_count: int):
    """
    Create a hook that replaces the first `img_count` token positions in the
    layer output with clean activations (dtype/device aligned).
    """
    @torch.no_grad()
    def _patch(module, inputs, output):
        out = unwrap_output(output)
        patched = out.clone()
        src = clean_act.to(patched.dtype).to(patched.device)
        if patched.size(1) < img_count or src.size(1) < img_count:
            k = min(img_count, patched.size(1), src.size(1))
        else:
            k = img_count
        patched[:, :k, :] = src[:, :k, :]
        if isinstance(output, tuple):
            lst = list(output)
            lst[0] = patched
            return tuple(lst)
        return patched

    return _patch


def group_patch_and_delta_both_metrics(
    model,
    processed_corrupt,
    layer_name: str,
    clean_act: torch.Tensor,
    img_count: int,
    base_scores: Dict[str, float],
    token_ids: List[int],
) -> Dict[str, float]:
    """
    Apply a single group (whole image-prefix) patch for the given layer and
    return deltas for both metrics: prob/logit.
    """
    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(
                make_group_patch_hook(clean_act, img_count)
            )
            break
    try:
        patched_logits = forward_logits(model, processed_corrupt)
        new_scores = last_token_scores(patched_logits, token_ids)
    finally:
        if handle:
            handle.remove()
    return {
        "prob": new_scores["prob"] - base_scores["prob"],
        "logit": new_scores["logit"] - base_scores["logit"],
    }


# ----------------- Runner -----------------
def run_experiment(
    adapter: BaseAdapter,
    data_path: str,
    clean_image_dir: str,
    corrupt_image_dir: str,
    layers_to_patch: Optional[List[str]],
    num_questions: int,
    output_dir: str,
):
    """
    Main entry: iterate over questions, cache clean activations, perform
    layerwise group patching on image prefix, and save per-question and
    averaged deltas (prob/logit).
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r") as f:
        data = json.load(f)
    selected = data[:num_questions]

    model, processor = adapter.load()
    tokenizer = adapter.tokenizer()
    layers = layers_to_patch or adapter.default_layers()
    img_token_count = adapter.image_token_count()

    L = len(layers)
    prob_accum = np.zeros(L, dtype=np.float64)
    logit_accum = np.zeros(L, dtype=np.float64)

    results_data = {
        "experiment_info": {
            "model_type": adapter.name,
            "model_id": getattr(adapter, "model_id", ""),
            "num_questions": num_questions,
            "data_path": data_path,
            "clean_image_dir": clean_image_dir,
            "corrupt_image_dir": corrupt_image_dir,
            "patch_scope": "layerwise_image_prefix",
            "image_token_count": int(img_token_count),
        },
        "per_question_results": [],
        "average_prob_changes": [],
        "average_logit_changes": [],
    }

    valid_count = 0

    for idx, q in enumerate(selected, 1):
        print(f"[{adapter.name}] {idx}/{num_questions} | question_id={q['question_id']}")
        clean_image_path = os.path.join(clean_image_dir, f"{q['imageId']}.png")
        corrupt_image_path = os.path.join(
            corrupt_image_dir, f"{q['question_id']}", f"{q['question_id']}.png"
        )
        question = q["question"]
        answer = q["answer"]

        try:
            clean_img = load_image(clean_image_path)
            corrupt_img = load_image(corrupt_image_path)
        except FileNotFoundError as e:
            print(f"  [Skip] {e}")
            continue

        prompt = adapter.prompt(question)
        proc_clean = adapter.preprocess(clean_img, prompt)
        proc_corrupt = adapter.preprocess(corrupt_img, prompt)

        try:
            token_ids = get_target_token_ids(tokenizer, answer)
        except Exception as e:
            print(f"  [Skip] tokenize answer failed: {e}")
            continue

        # Baseline scores on the corrupt input
        base_logits = forward_logits(model, proc_corrupt)
        base_scores = last_token_scores(base_logits, token_ids)

        # Cache clean activations for all target layers
        clean_acts = cache_clean_activations(model, layers, proc_clean)

        q_record = {
            "question_id": q["question_id"],
            "question": question,
            "correct_answer": answer,
            "image_id": q["imageId"],
            "layer_results": [],
        }

        # Per-layer: patch the first img_token_count positions and measure deltas
        for li, layer_name in enumerate(layers):
            clean_act = clean_acts[layer_name]
            deltas = group_patch_and_delta_both_metrics(
                model=model,
                processed_corrupt=proc_corrupt,
                layer_name=layer_name,
                clean_act=clean_act,
                img_count=img_token_count,
                base_scores=base_scores,
                token_ids=token_ids,
            )
            prob_accum[li] += deltas["prob"]
            logit_accum[li] += deltas["logit"]

            q_record["layer_results"].append(
                {
                    "layer_index": li,
                    "layer_name": layer_name,
                    "prob_delta": float(deltas["prob"]),
                    "logit_delta": float(deltas["logit"]),
                }
            )

        results_data["per_question_results"].append(q_record)
        valid_count += 1

    valid_count = max(1, valid_count)
    avg_prob = (prob_accum / valid_count).astype(np.float64)
    avg_logit = (logit_accum / valid_count).astype(np.float64)

    results_data["average_prob_changes"] = avg_prob.tolist()
    results_data["average_logit_changes"] = avg_logit.tolist()

    # Save results: JSON + NPY
    json_path = os.path.join(output_dir, f"{adapter.name}_layerwise_patch_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"[Save] JSON -> {json_path}")

    np.save(os.path.join(output_dir, f"{adapter.name}_avg_prob_changes.npy"), avg_prob)
    np.save(os.path.join(output_dir, f"{adapter.name}_avg_logit_changes.npy"), avg_logit)
    print(f"[Save] NPY -> avg_prob_changes / avg_logit_changes")

    return results_data


# ----------------- Entrypoint -----------------
if __name__ == "__main__":
    # Adjust paths as needed.
    if MODEL_TYPE == "llava":
        adapter = LLaVAAdapter(LLAVA_MODEL_ID, image_tokens=LLAVA_IMAGE_TOKENS)
        data_path = "../data/GQA/Object_Level/Animal/Animal.json"
        clean_image_dir = "../data/GQA/Object_Level/Animal/clean"
        corrupt_image_dir = "../data/GQA/Object_Level/Animal/corrupt"
        output_dir = "../data/GQA/Object_Level/Animal/LLAVALayerwiseImpact"
        layers_to_patch = [f"language_model.model.layers.{i}" for i in range(32)]  # whole layer

    elif MODEL_TYPE == "instructblip":
        adapter = InstructBLIPAdapter(INSTRUCTBLIP_MODEL_ID, use_4bit=True)
        data_path = "../data/GQA/Object_Level/Animal/Animal.json"
        clean_image_dir = "../data/GQA/Object_Level/Animal/clean"
        corrupt_image_dir = "../data/GQA/Object_Level/Animal/corrupt"
        output_dir = "../data/GQA/Object_Level/Animal/BLIPLayerwiseImpact"
        layers_to_patch = [f"language_model.model.layers.{i}" for i in range(32)]  # whole layer

    else:
        raise ValueError("MODEL_TYPE must be 'llava' or 'instructblip'")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting layerwise image-prefix patching ({MODEL_TYPE}) ...")
    run_experiment(
        adapter=adapter,
        data_path=data_path,
        clean_image_dir=clean_image_dir,
        corrupt_image_dir=corrupt_image_dir,
        layers_to_patch=layers_to_patch,
        num_questions=NUM_QUESTIONS,
        output_dir=output_dir,
    )

