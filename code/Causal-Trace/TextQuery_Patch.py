#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
from typing import List, Dict, Optional
from PIL import Image
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor

# ====================== Config ======================
MODEL_TYPE = "llava"  # "llava" | "instructblip"

# HuggingFace model IDs (online)
LLAVA_MODEL_ID = "liuhaotian/llava-v1.5-7b"
INSTRUCTBLIP_MODEL_ID = "Salesforce/instructblip-vicuna-7b"

# Text starts after these many "image/prefix" tokens

LLAVA_IMAGE_TOKENS = 576

INSTRUCTBLIP_FALLBACK_QTOK = 32

NUM_QUESTIONS = 1000
# ====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------- I/O & helpers -----------------
def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


@torch.no_grad()
def forward_logits(model, processed_inputs) -> torch.Tensor:
    """Single forward pass; returns logits [B, T, V]."""
    outputs = model(**processed_inputs)
    return outputs.logits


def unwrap_output(output):
    """Normalize module output to a Tensor (hidden_states)."""
    if isinstance(output, tuple):
        return output[0]
    return output


def get_target_token_ids(tokenizer, answer: str) -> List[int]:
    """
    Tokenize the answer robustly into vocab ids without special tokens.
    Falls back to lower-cased version if needed.
    """
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer.strip().lower(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer '{answer}' cannot be tokenized to any ids.")
    return ids


def last_token_scores(logits: torch.Tensor, token_ids: List[int]) -> Dict[str, float]:
    """
    Return both metrics at the last time step:
      - prob : sum of softmax probabilities over target ids
      - logit: mean of raw logits over target ids
    """
    last = logits[:, -1, :]  # [B, V]
    probs = torch.softmax(last, dim=-1)
    prob_val = float(sum(probs[0, t].item() for t in token_ids))
    logit_val = float(torch.stack([last[0, t] for t in token_ids]).mean().item())
    return {"prob": prob_val, "logit": logit_val}


# ----------------- Adapters -----------------
class BaseAdapter:
    """Abstract adapter for model/processor and model-specific logic."""
    name = "base"

    def load(self):
        raise NotImplementedError

    def preprocess(self, image: Image.Image, text: str):
        raise NotImplementedError

    def prompt(self, question: str) -> str:
        raise NotImplementedError

    def tokenizer(self):
        return self.processor.tokenizer

    def text_start_index(self) -> int:
        """Index where TEXT tokens begin (i.e., after image-prefix tokens)."""
        raise NotImplementedError

    def default_layers(self) -> List[str]:
        """Whole Transformer blocks by default; override if needed."""
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

    def text_start_index(self) -> int:
        # TEXT tokens start right after the image-prefix tokens
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

    def text_start_index(self) -> int:
        """
        TEXT begins after Q-Former query tokens (image queries).
        """
        qconf = getattr(getattr(self.model, "config", None), "qformer_config", None)
        qtok = INSTRUCTBLIP_FALLBACK_QTOK
        if qconf and hasattr(qconf, "num_query_tokens"):
            qtok = int(qconf.num_query_tokens)
        return qtok


# ----------------- Activation cache & patch -----------------
def cache_clean_activations(
    model, layers_to_patch: List[str], processed_clean
) -> Dict[str, torch.Tensor]:
    """Run a clean forward pass and cache layer outputs."""
    clean_acts = {}
    hooks = []

    def make_hook(layer_name: str):
        def _hook(module, inputs, output):
            out = unwrap_output(output)
            if layer_name not in clean_acts:
                clean_acts[layer_name] = out.detach().clone()
        return _hook

    for layer_name in layers_to_patch:
        for name, module in model.named_modules():
            if name == layer_name:
                hooks.append(module.register_forward_hook(make_hook(layer_name)))
                break

    _ = forward_logits(model, processed_clean)

    for h in hooks:
        h.remove()
    return clean_acts


def make_text_patch_hook(clean_act: torch.Tensor, start_idx: int):
    """
    Hook that replaces TEXT token positions [start_idx: ] with clean activations.
    Handles dtype/device alignment and tuple outputs.
    """
    @torch.no_grad()
    def _patch(module, inputs, output):
        out = unwrap_output(output)
        patched = out.clone()
        src = clean_act.to(patched.dtype).to(patched.device)
        T = min(patched.size(1), src.size(1))
        if start_idx < T:
            patched[:, start_idx:T, :] = src[:, start_idx:T, :]
        if isinstance(output, tuple):
            lst = list(output)
            lst[0] = patched
            return tuple(lst)
        return patched
    return _patch


def group_patch_text_and_deltas(
    model,
    processed_corrupt,
    layer_name: str,
    clean_act: torch.Tensor,
    start_idx: int,
    base_scores: Dict[str, float],
    token_ids: List[int],
) -> Dict[str, float]:
    """
    Replace TEXT-token hidden states for a single layer, compute both deltas:
    prob_delta and logit_delta.
    """
    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(
                make_text_patch_hook(clean_act, start_idx)
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
    For each question:
      1) Build clean/corrupt inputs.
      2) Cache clean activations at target layers.
      3) For each layer, replace TEXT tokens [start_idx:] in the corrupt run with clean,
         then measure deltas for prob/logit on the final position.
    Saves:
      - JSON with per-question layer results and averaged curves
      - NPY arrays: average_prob_changes.npy / average_logit_changes.npy
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r") as f:
        data = json.load(f)
    selected = data[:num_questions]

    model, processor = adapter.load()
    tokenizer = adapter.tokenizer()
    layers = layers_to_patch or adapter.default_layers()
    text_start_idx = adapter.text_start_index()

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
            "patch_scope": "layerwise_text_tokens",
            "text_start_index": int(text_start_idx),
        },
        "per_question_results": [],
        "average_prob_changes": [],
        "average_logit_changes": [],
    }

    valid = 0

    for i, q in enumerate(selected, 1):
        print(f"[{adapter.name}] {i}/{num_questions} | question_id={q['question_id']}")
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

        # Baseline on corrupt input
        base_logits = forward_logits(model, proc_corrupt)
        base_scores = last_token_scores(base_logits, token_ids)

        # Cache clean activations across target layers
        clean_acts = cache_clean_activations(model, layers, proc_clean)

        q_rec = {
            "question_id": q["question_id"],
            "question": question,
            "correct_answer": answer,
            "image_id": q["imageId"],
            "layer_results": [],
        }

        for li, layer_name in enumerate(layers):
            clean_act = clean_acts[layer_name]
            deltas = group_patch_text_and_deltas(
                model=model,
                processed_corrupt=proc_corrupt,
                layer_name=layer_name,
                clean_act=clean_act,
                start_idx=text_start_idx,
                base_scores=base_scores,
                token_ids=token_ids,
            )
            prob_accum[li] += deltas["prob"]
            logit_accum[li] += deltas["logit"]

            q_rec["layer_results"].append(
                {
                    "layer_index": li,
                    "layer_name": layer_name,
                    "prob_delta": float(deltas["prob"]),
                    "logit_delta": float(deltas["logit"]),
                }
            )

        results_data["per_question_results"].append(q_rec)
        valid += 1

    valid = max(1, valid)
    avg_prob = (prob_accum / valid).astype(np.float64)
    avg_logit = (logit_accum / valid).astype(np.float64)

    results_data["average_prob_changes"] = avg_prob.tolist()
    results_data["average_logit_changes"] = avg_logit.tolist()

    # Save to disk
    json_path = os.path.join(output_dir, f"{adapter.name}_text_patch_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"[Save] JSON -> {json_path}")

    np.save(os.path.join(output_dir, f"{adapter.name}_average_prob_changes.npy"), avg_prob)
    np.save(os.path.join(output_dir, f"{adapter.name}_average_logit_changes.npy"), avg_logit)
    print("[Save] NPY -> average_prob_changes / average_logit_changes")

    return results_data


# ----------------- Entrypoint -----------------
if __name__ == "__main__":
    # Set paths as needed
    if MODEL_TYPE == "llava":
        adapter = LLaVAAdapter(LLAVA_MODEL_ID, image_tokens=LLAVA_IMAGE_TOKENS)
        data_path = "../data/GQA/Object_Level/Animal/Animal.json"
        clean_image_dir = "../data/GQA/Object_Level/Animal/clean"
        corrupt_image_dir = "../data/GQA/Object_Level/Animal/corrupt"
        output_dir = "../data/GQA/Object_Level/Animal/LLAVALayerwiseImpact"
        layers_to_patch = [f"language_model.model.layers.{i}" for i in range(32)]  # whole blocks

    elif MODEL_TYPE == "instructblip":
        adapter = InstructBLIPAdapter(INSTRUCTBLIP_MODEL_ID, use_4bit=True)
        data_path = "../data/GQA/Object_Level/Animal/Animal.json"
        clean_image_dir = "../data/GQA/Object_Level/Animal/clean"
        corrupt_image_dir = "../data/GQA/Object_Level/Animal/corrupt"
        output_dir = "../data/GQA/Object_Level/Animal/BLIPLayerwiseImpact"
        layers_to_patch = [f"language_model.model.layers.{i}" for i in range(32)]  # whole blocks

    else:
        raise ValueError("MODEL_TYPE must be 'llava' or 'instructblip'")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting layerwise TEXT-token patching ({MODEL_TYPE}) ...")
    run_experiment(
        adapter=adapter,
        data_path=data_path,
        clean_image_dir=clean_image_dir,
        corrupt_image_dir=corrupt_image_dir,
        layers_to_patch=layers_to_patch,
        num_questions=NUM_QUESTIONS,
        output_dir=output_dir,
    )

