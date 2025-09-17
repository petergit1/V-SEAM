#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import argparse
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------- I/O helpers ---------------------
def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


@torch.no_grad()
def forward_logits(model, processed_inputs):
    out = model(**processed_inputs)
    return out.logits  # [B, T, V]


def unwrap_output(output):
    """Normalize module outputs to a Tensor (pull hidden_states when tuple)."""
    if isinstance(output, tuple):
        return output[0]
    return output


def get_target_token_ids(tokenizer, answer: str) -> List[int]:
    """
    Robust tokenization into vocab ids without special tokens.
    Falls back to lowercased form if needed.
    """
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer.strip().lower(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer '{answer}' cannot be tokenized to any ids.")
    return ids


def last_token_scores(logits: torch.Tensor, token_ids: List[int]) -> Dict[str, float]:
    """
    Return both metrics on the last time step:
      - prob : sum of softmax probabilities over target ids
      - logit: mean of raw logits over target ids
    """
    last = logits[:, -1, :]  # [B, V]
    probs = torch.softmax(last, dim=-1)
    prob_val = float(sum(probs[0, t].item() for t in token_ids))
    logit_val = float(torch.stack([last[0, t] for t in token_ids]).mean().item())
    return {"prob": prob_val, "logit": logit_val}


# --------------------- Adapters ---------------------
class BaseAdapter:
    """Abstract adapter for model/processor specifics."""
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
        """Index where TEXT tokens begin (positions < this are image/prefix)."""
        raise NotImplementedError

    def attn_layer_names(self, num_layers: int) -> List[str]:
        """Return full module names to hook self_attn for each layer."""
        return [f"language_model.model.layers.{i}.self_attn" for i in range(num_layers)]


class LLaVAAdapter(BaseAdapter):
    name = "llava"

    def __init__(self, model_id: str, image_tokens: int, num_layers: int = 32):
        self.model_id = model_id
        self._image_tokens = image_tokens
        self._num_layers = num_layers

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
        return int(self._image_tokens)

    def attn_layer_names(self, num_layers: int) -> List[str]:
        return [f"language_model.model.layers.{i}.self_attn" for i in range(num_layers)]


class InstructBLIPAdapter(BaseAdapter):
    name = "instructblip"

    def __init__(self, model_id: str, use_4bit: bool = True, num_layers: int = 32, qtok_fallback: int = 32):
        self.model_id = model_id
        self.use_4bit = use_4bit
        self._num_layers = num_layers
        self.qtok_fallback = qtok_fallback

    def load(self):
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor, BitsAndBytesConfig
        quant_kwargs = {}
        if self.use_4bit:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        self.processor = InstructBlipProcessor.from_pretrained(self.model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(self.model_id, **quant_kwargs)
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
        # Text begins after Q-Former queries
        qconf = getattr(getattr(self.model, "config", None), "qformer_config", None)
        if qconf and hasattr(qconf, "num_query_tokens"):
            return int(qconf.num_query_tokens)
        return int(self.qtok_fallback)

    def attn_layer_names(self, num_layers: int) -> List[str]:
        return [f"language_model.model.layers.{i}.self_attn" for i in range(num_layers)]


# --------------------- Caching & Patching (ATTN output) ---------------------
def cache_clean_attn_outputs(
    model,
    attn_layers: List[str],
    processed_clean
) -> Dict[str, torch.Tensor]:
    """
    Run a clean forward once and cache self_attn outputs for each target layer.
    Returns dict: layer_name -> Tensor [B, T, D]
    """
    cache = {}
    hooks = []

    def make_hook(layer_name: str):
        def _hook(module, inputs, output):
            out = unwrap_output(output)  # [B, T, D]
            cache[layer_name] = out.detach().clone()
        return _hook

    for layer_name in attn_layers:
        for name, module in model.named_modules():
            if name == layer_name:
                hooks.append(module.register_forward_hook(make_hook(layer_name)))
                break

    _ = forward_logits(model, processed_clean)

    for h in hooks:
        h.remove()

    return cache


def patch_one_attn_position_and_delta(
    model,
    processed_corrupt,
    layer_name: str,
    pos_idx: int,
    clean_attn_cache: Dict[str, torch.Tensor],
    base_scores: Dict[str, float],
    token_ids: List[int]
) -> Dict[str, float]:
    """
    Replace the ENTIRE self_attn output at a specific token position for a single layer,
    then compute deltas on the final-step scores.
    """
    def _patch_fn(module, inputs, output):
        out = unwrap_output(output)  # [B, T, D]
        patched = out.clone()
        src = clean_attn_cache[layer_name].to(patched.dtype).to(patched.device)
        T = min(patched.size(1), src.size(1))
        if pos_idx < T:
            patched[:, pos_idx, :] = src[:, pos_idx, :]
        if isinstance(output, tuple):
            lst = list(output)
            lst[0] = patched
            return tuple(lst)
        return patched

    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(_patch_fn)
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


# --------------------- Core (per-question scan) ---------------------
def tokenwise_attention_scan_for_question(
    model,
    processor,
    adapter: BaseAdapter,
    attn_layers: List[str],
    clean_img: Image.Image,
    corrupt_img: Image.Image,
    question_text: str,
    answer_text: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform token-wise attention-output patch scan for a single (clean, corrupt) pair.

    Returns:
      prob_mat  : [num_layers, text_len]
      logit_mat : [num_layers, text_len]
      text_len  : int (number of text tokens considered)
    """
    prompt = adapter.prompt(question_text)
    proc_clean   = adapter.preprocess(clean_img, prompt)
    proc_corrupt = adapter.preprocess(corrupt_img, prompt)

    token_ids = get_target_token_ids(adapter.tokenizer(), answer_text)

    # Baseline (corrupt)
    base_logits = forward_logits(model, proc_corrupt)
    base_scores = last_token_scores(base_logits, token_ids)

    # Cache clean self_attn outputs
    clean_attn_cache = cache_clean_attn_outputs(model, attn_layers, proc_clean)

    # Determine text token span
    input_ids = proc_clean["input_ids"]  # [1, T]
    T_total = int(input_ids.shape[1])
    start = int(adapter.text_start_index())
    start = max(0, min(start, T_total))
    text_positions = list(range(start, T_total))
    text_len = len(text_positions)

    L = len(attn_layers)
    prob_mat = np.zeros((L, text_len), dtype=np.float64)
    logit_mat = np.zeros((L, text_len), dtype=np.float64)

    for li, layer_name in enumerate(attn_layers):
        for ti, pos in enumerate(text_positions):
            deltas = patch_one_attn_position_and_delta(
                model=model,
                processed_corrupt=proc_corrupt,
                layer_name=layer_name,
                pos_idx=pos,
                clean_attn_cache=clean_attn_cache,
                base_scores=base_scores,
                token_ids=token_ids
            )
            prob_mat[li, ti]  = deltas["prob"]
            logit_mat[li, ti] = deltas["logit"]

    return prob_mat, logit_mat, text_len


# --------------------- Averaging across questions ---------------------
def accumulate_avg_matrices(
    avg_prob: Optional[np.ndarray],
    avg_logit: Optional[np.ndarray],
    new_prob: np.ndarray,
    new_logit: np.ndarray,
    q_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Online mean over questions with right-padding to the current max text length.
    All matrices are [L, T_text].
    """
    if avg_prob is None:
        return new_prob.copy(), new_logit.copy()

    L_old, T_old = avg_prob.shape
    L_new, T_new = new_prob.shape
    L = max(L_old, L_new)
    T = max(T_old, T_new)

    def pad(mat, L, T):
        Lm, Tm = mat.shape
        out = np.zeros((L, T), dtype=mat.dtype)
        out[:Lm, :Tm] = mat
        return out

    avg_prob = pad(avg_prob, L, T)
    avg_logit = pad(avg_logit, L, T)
    new_prob = pad(new_prob, L, T)
    new_logit = pad(new_logit, L, T)

    q = q_index + 1  # q_index starts at 0
    avg_prob  = ((q - 1) * avg_prob  + new_prob ) / q
    avg_logit = ((q - 1) * avg_logit + new_logit) / q
    return avg_prob, avg_logit


# --------------------- Top-level experiment ---------------------
def run_experiment(
    model_type: str,
    model_id: Optional[str],
    data_path: str,
    clean_dir: str,
    corrupt_dir: str,
    output_dir: str,
    num_questions: int,
    image_tokens_llava: int,
    qtok_fallback_instruct: int,
    num_layers: int
):
    """
    Iterate dataset, perform token-wise self-attention-output patch scan, and save averages.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r") as f:
        dataset = json.load(f)
    selected = dataset[:num_questions]

    # Build adapter & load model
    if model_type.lower() == "llava":
        adapter = LLaVAAdapter(
            model_id=model_id or "liuhaotian/llava-v1.5-7b",
            image_tokens=image_tokens_llava,
            num_layers=num_layers
        )
    elif model_type.lower() == "instructblip":
        adapter = InstructBLIPAdapter(
            model_id=model_id or "Salesforce/instructblip-vicuna-7b",
            use_4bit=True,
            num_layers=num_layers,
            qtok_fallback=qtok_fallback_instruct
        )
    else:
        raise ValueError("model_type must be 'llava' or 'instructblip'")

    model, processor = adapter.load()
    _ = adapter.tokenizer()  # ensure tokenizer exists
    attn_layers = adapter.attn_layer_names(num_layers)

    avg_prob, avg_logit = None, None
    valid = 0

    for qi, q in enumerate(selected):
        print(f"[{adapter.name}] {qi+1}/{len(selected)} | qid={q['question_id']}")
        clean_path   = os.path.join(clean_dir,   f"{q['imageId']}.png")
        corrupt_path = os.path.join(corrupt_dir, f"{q['question_id']}", f"{q['question_id']}.png")

        try:
            clean_img = load_image(clean_path)
            corrupt_img = load_image(corrupt_path)
        except FileNotFoundError as e:
            print(f"  [Skip] {e}")
            continue

        try:
            prob_mat, logit_mat, text_len = tokenwise_attention_scan_for_question(
                model=model,
                processor=processor,
                adapter=adapter,
                attn_layers=attn_layers,
                clean_img=clean_img,
                corrupt_img=corrupt_img,
                question_text=q["question"],
                answer_text=q["answer"]
            )
        except Exception as e:
            print(f"  [Error] qid={q['question_id']} -> {e}")
            continue

        avg_prob, avg_logit = accumulate_avg_matrices(avg_prob, avg_logit, prob_mat, logit_mat, valid)
        valid += 1

    if valid == 0:
        print("[Warn] No valid samples processed; nothing to save.")
        return

    # Save averaged results
    np.save(os.path.join(output_dir, "avg_prob_deltas.npy"),  avg_prob)
    np.save(os.path.join(output_dir, "avg_logit_deltas.npy"), avg_logit)

    meta = {
        "model_type": adapter.name,
        "model_id": getattr(adapter, "model_id", ""),
        "num_layers": num_layers,
        "num_questions_requested": num_questions,
        "num_questions_processed": valid,
        "text_start_index_hint": adapter.text_start_index(),
        "avg_prob_deltas_shape": list(avg_prob.shape),
        "avg_logit_deltas_shape": list(avg_logit.shape),
        "attn_layers": attn_layers,
        "data_path": data_path,
        "clean_dir": clean_dir,
        "corrupt_dir": corrupt_dir,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Save] avg_prob_deltas.npy / avg_logit_deltas.npy and summary.json saved to {output_dir}")


# --------------------- CLI ---------------------
def build_cli():
    p = argparse.ArgumentParser(description="Token-wise self-attention output patching for LLaVA / InstructBLIP.")
    p.add_argument("--model_type", type=str, required=True, choices=["llava", "instructblip"],
                   help="Model family to use.")
    p.add_argument("--model_id", type=str, default=None,
                   help="HuggingFace model id (override default).")
    p.add_argument("--data_path", type=str, required=True,
                   help="JSON list with {question_id, imageId, question, answer}.")
    p.add_argument("--clean_dir", type=str, required=True,
                   help="Dir with clean images named <imageId>.png")
    p.add_argument("--corrupt_dir", type=str, required=True,
                   help="Dir with corrupt images in subfolders <question_id>/<question_id>.png")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Dir to save NPY/JSON results.")
    p.add_argument("--num_questions", type=int, default=1000,
                   help="Number of samples to process.")
    p.add_argument("--num_layers", type=int, default=32,
                   help="How many transformer layers to scan (from 0).")
    p.add_argument("--image_tokens_llava", type=int, default=576,
                   help="For LLaVA: number of image-prefix tokens (text starts after this).")
    p.add_argument("--qtok_fallback_instruct", type=int, default=32,
                   help="For InstructBLIP: fallback Q-Former query count if not detectable.")
    return p


if __name__ == "__main__":
    args = build_cli().parse_args()
    run_experiment(
        model_type=args.model_type,
        model_id=args.model_id,
        data_path=args.data_path,
        clean_dir=args.clean_dir,
        corrupt_dir=args.corrupt_dir,
        output_dir=args.output_dir,
        num_questions=args.num_questions,
        image_tokens_llava=args.image_tokens_llava,
        qtok_fallback_instruct=args.qtok_fallback_instruct,
        num_layers=args.num_layers,
    )
