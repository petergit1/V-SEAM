#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torchvision.transforms.functional import to_pil_image
import copy
from collections import OrderedDict, defaultdict
import torch
import numpy as np
from PIL import Image
import pickle
from typing import Any, Optional, Tuple, Union, Sequence, Dict, List, Iterable
import contextlib
import inspect
from rapidfuzz import process, fuzz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================== Generic text utilities ======================

def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def clean_and_combine_triplets(pos_triplet: str, neg_triplet: str, keep: int = 3) -> List[str]:
    def _split(s: str) -> List[str]:
        s = s.replace('[', '').replace(']', '').replace("'", "").strip()
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return parts[:keep]

    combined = _split(pos_triplet) + _split(neg_triplet)
    return _unique_preserve_order(combined)


def generate_toi_matches(
    sentence: str,
    tokenizer,
    terms: Iterable[str],
    include_special: Iterable[str] = ('?', 'or'),
) -> Tuple[Dict[str, Optional[int]], List[int]]:
    """
    Generic token-of-interest (TOI) matcher.
    - sentence: full prompt string
    - tokenizer: HuggingFace tokenizer (e.g., processor.tokenizer)
    - terms: iterable of strings you want to locate in token space (e.g., {'happy','sad'} or SVO terms)
    - include_special: extra symbols/words to record if present (optional)

    Returns:
        matches: dict term -> end_index_of_match (int) or None if not found
        index_list: sorted list of all found indices
    """
    # Tokenize sentence (no special tokens to keep indices aligned to plain text)
    enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    token_ids = enc["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Helper: exact sliding-window match for tokenized term
    def _match_term(term: str) -> Optional[int]:
        # Leading space helps enforce word boundary for SP/BPE tokenizers
        term_tokens = tokenizer.tokenize(" " + term)
        if not term_tokens:
            return None
        L = len(term_tokens)
        for i in range(0, len(tokens) - L + 1):
            if tokens[i:i + L] == term_tokens:
                # Return the *last* token index of the matched span for consistency
                return i + L - 1
        return None

    # Normalize tokens for fuzzy search (remove common prefixes)
    def _norm(tok: str) -> str:
        # Strip common subword prefixes without assuming model family
        return tok.replace('▁', '').replace('Ġ', '').replace('##', '')

    matches: Dict[str, Optional[int]] = {}

    # 1) Exact sliding-window match
    remaining: List[str] = []
    for term in terms:
        term = str(term).strip()
        idx = _match_term(term)
        if idx is not None:
            matches[term] = idx
        else:
            remaining.append(term)

    # 2) Record optional specials (e.g., '?' / 'or')
    for sp in include_special:
        sp_tok = tokenizer.tokenize(sp)
        found = None
        if sp_tok:
            # single or multi-token special
            Ls = len(sp_tok)
            for i in range(0, len(tokens) - Ls + 1):
                if tokens[i:i + Ls] == sp_tok:
                    found = i + Ls - 1
                    break
        matches[sp] = found

    # 3) Fuzzy fallback for unmatched terms (use non-matched token positions only for stability)
    used_positions = set(v for v in matches.values() if isinstance(v, int))
    candidate_positions = [i for i in range(len(tokens)) if i not in used_positions]
    candidate_token_texts = [_norm(t) for i, t in enumerate(tokens) if i in candidate_positions]

    for term in remaining:
        if not candidate_positions:
            matches[term] = None
            continue
        best = process.extractOne(term, candidate_token_texts, scorer=fuzz.WRatio)
        if best and best[1] > 55:
            chosen_text = best[0]
            # Map back to index while avoiding already used positions
            for i in candidate_positions:
                if _norm(tokens[i]) == chosen_text:
                    matches[term] = i
                    used_positions.add(i)
                    candidate_positions = [j for j in candidate_positions if j != i]
                    break
        else:
            matches[term] = None


    sorted_items = sorted([(k, v) for k, v in matches.items() if isinstance(v, int)], key=lambda kv: kv[1])
    index_list = [v for _, v in sorted_items]

    return matches, index_list


# ====================== Module tracing utilities ======================

class StopForward(Exception):
    """Internal control-flow exception to stop forward after a retained layer."""
    pass


def recursive_copy(x, clone: bool = None, detach: bool = None, retain_grad: bool = None):
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v, clone, detach, retain_grad) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v, clone, detach, retain_grad) for v in x])
    else:
        raise TypeError(f"Unknown type {type(x)} cannot be broken into tensors.")


def subsequence(
    sequential: torch.nn.Sequential,
    first_layer: Optional[str] = None,
    last_layer: Optional[str] = None,
    after_layer: Optional[str] = None,
    upto_layer: Optional[str] = None,
    single_layer: Optional[str] = None,
    share_weights: bool = False,
):
    assert (single_layer is None) or (first_layer is last_layer is after_layer is upto_layer is None)
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [
        None if d is None else d.split(".") for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )


def hierarchical_subsequence(
    sequential: torch.nn.Sequential,
    first, last, after, upto,
    share_weights: bool = False,
    depth: int = 0
):
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    if not isinstance(sequential, torch.nn.Sequential):
        path = ".".join((first or last or after or upto)[:depth] or ["arg"])
        raise TypeError(f"{path} not Sequential")

    including_children = (first is None) and (after is None)
    included_children = OrderedDict()

    (F, FN), (L, LN), (A, AN), (U, UN) = [
        ((d[depth], None if len(d) == depth + 1 else d) if d is not None else (None, None))
        for d in [first, last, after, upto]
    ]

    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False

        if including_children:
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            chosen = hierarchical_subsequence(
                layer,
                first=FR, last=LR, after=AR, upto=UR,
                share_weights=share_weights, depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen

        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True

    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))

    if not len(included_children) and depth > 0:
        return None

    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result


def set_requires_grad(requires_grad: bool, *models: Any):
    for model in models:
        if isinstance(model, torch.nn.Module):
            for p in model.parameters():
                p.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            raise TypeError(f"unknown type {type(model)}")


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def replace_module(model: torch.nn.Module, name: str, new_module: torch.nn.Module):
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
    else:
        attr_name = name
    setattr(model, attr_name, new_module)


def invoke_with_optional_args(fn, *args, **kwargs):
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (0 if not argspec.defaults else len(argspec.defaults))
    # pass positional that match names first
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n]); used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos]); used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(None if i < defaulted_pos else argspec.defaults[i - defaulted_pos])

    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(argspec.args[u] for u in unmatched_pos if u < defaulted_pos)
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")

    pass_kw = {
        k: v for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)


class Trace(contextlib.AbstractContextManager):
    """
    Hook a layer to retain/optionally edit its (input, output) during forward.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        layer: Optional[str] = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output=None,
        stop: bool = False,
    ):
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)

        retainer = self

        def retain_hook(m, inputs, output):
            if retain_input:
                retainer.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone, detach=detach, retain_grad=False
                )
            if edit_output:
                output = invoke_with_optional_args(edit_output, output=output, layer=self.layer)
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, _type, value, traceback):
        self.close()
        if self.stop and _type and issubclass(_type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    Retain multiple layers in one context.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        layers: Sequence[str],
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output=None,
        stop: bool = False,
    ):
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module, layer=layer,
                retain_output=retain_output, retain_input=retain_input,
                clone=clone, detach=detach, retain_grad=retain_grad,
                edit_output=edit_output, stop=(stop and is_last),
            )

    def __enter__(self):
        return self

    def __exit__(self, _type, value, traceback):
        self.close()
        if self.stop and _type and issubclass(_type, StopForward):
            return True

    def close(self):
        for _, trace in reversed(self.items()):
            trace.close()


# ====================== Model-specific helpers & causal patching ======================

def layername(model: torch.nn.Module, num: int, block_name: str, kind: Optional[Union[str, Tuple[str]]] = None) -> str:
    """
    Construct a layer path for common VLM stacks without embedding dataset assumptions.
    """
    base_paths = {
        "vision_model": "vision_model",
        "text_encoder": "text_encoder.encoder.layer",
        "text_decoder": "text_decoder.bert.encoder.layer",
        "language_model": "language_model.model.layers",
    }

    special_kinds = {
        "embed": {
            "vision_model": "vision_model.embeddings",
            "text_encoder": "text_encoder.embeddings",
            "text_decoder": "text_decoder.bert.embeddings",
            "language_model": "language_model.model.embed_tokens",
        },
        "cls": {
            "text_decoder": "text_decoder.cls",
        },
        "crossattention_layernorm": {
            "text_encoder": f"text_encoder.encoder.layer.{num}.crossattention.output.LayerNorm"
        },
        "crossattention_block": {
            "text_encoder": f"text_encoder.encoder.layer.{num}.crossattention.self"
        },
        "attention_block": {
            "text_encoder": f"text_encoder.encoder.layer.{num}.attention.self",
            "language_model": f"language_model.model.layers.{num}.self_attn",
        },
        "mlp_block": {
            "language_model": f"language_model.model.layers.{num}.mlp",
        },
    }

    if kind in special_kinds and block_name in special_kinds[kind]:  # type: ignore
        return special_kinds[kind][block_name]  # type: ignore

    base_path = base_paths.get(block_name)
    if base_path is None:
        raise ValueError(f"Unknown transformer architecture: '{block_name}'")

    layer_path = f"{base_path}.{num}"
    if kind not in ["embed", "cls", None, "attention_block", "mlp_block"]:
        # 'kind' might be a tuple in some callers; keep the first element
        k = kind[0] if isinstance(kind, (tuple, list)) else kind
        layer_path += f".{k}"
    return layer_path


def predict_from_input(model, pixel_values, input_ids, attention_mask=None):
    return model.forward(pixel_values=pixel_values.to(device), input_ids=input_ids.to(device))


def result_gen(output, return_p: bool = False):
    out = output['decoder_logits']
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def result_gen_min(output, return_p: bool = False):
    out = output['decoder_logits']
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.min(probs, dim=1)
    return preds, p


def decoding(pred_prob, processor) -> str:
    preds = pred_prob[0]
    ans = []
    for i in range(preds.size(0)):
        single_pred = preds[i]
        decoded_answer = processor.decode(single_pred, skip_special_tokens=True)
        ans.append(decoded_answer)
    return ans[0]


def get_svo_tokens(processor, correct_text: str, incorrect_text: str) -> Tuple[int, int]:
    """
    Utility to convert (correct, incorrect) answer strings into their *last-token* ids,
    which is often how classification heads are read in generative models.
    """
    correct_ids = processor(text=correct_text).input_ids.squeeze()
    incorrect_ids = processor(text=incorrect_text).input_ids.squeeze()
    return int(correct_ids[-1]), int(incorrect_ids[-1])


def trace_with_patch(
    model,
    processor,
    constant_input,          # image or text counterpart (depends on mode)
    clean_input,             # the clean modality (image or text)
    corrupt_input,           # the corrupted modality
    svo_tokens: Sequence[int],
    mode: str,               # 'image' or 'text'
    states_to_patch: List[Tuple[int, str]],  # [(token_index, layer_path), ...]
    attn_head: Optional[int] = None,
    knockout: Optional[Any] = None,
):
    """
    Core causal patcher: run clean vs corrupt; restore specified hidden states
    (or attention head vectors) from clean into corrupt at given layers/tokens.
    """

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(int(t))

    # Prepare batch inputs
    if mode == 'image':
        processed_inputs_clean = processor(images=clean_input, text=constant_input, return_tensors='pt', padding=True)
        processed_inputs_corr  = processor(images=corrupt_input, text=constant_input, return_tensors='pt', padding=True)
    elif mode == 'text':
        processed_inputs_clean = processor(images=constant_input, text=clean_input, return_tensors='pt', padding=True)
        processed_inputs_corr  = processor(images=constant_input, text=corrupt_input, return_tensors='pt', padding=True)
    else:
        raise ValueError("Invalid mode specified. Use 'image' or 'text'.")

    pixel_val_clean = processed_inputs_clean.pixel_values.to(device)
    pixel_val_corr  = processed_inputs_corr.pixel_values.to(device)
    input_ids_clean = processed_inputs_clean.input_ids.to(device)
    input_ids_corr  = processed_inputs_corr.input_ids.to(device)
    attention_mask_clean = processed_inputs_clean.attention_mask.to(device)
    attention_mask_corr  = processed_inputs_corr.attention_mask.to(device)

    # Run clean forward to cache donor states
    if attn_head is not None:
        outputs_exp_clean = model(
            input_ids=input_ids_clean, pixel_values=pixel_val_clean,
            attention_mask=attention_mask_clean, output_attentions=True
        )
        clean_states = outputs_exp_clean.attentions
        # attentions: typically List[Tensor] with shape [B, n_heads, seq, seq] or model-specific
        # NOTE: patching attention outputs is model-dependent; we follow original logic for continuity
    else:
        outputs_exp_clean = model(
            input_ids=input_ids_clean, pixel_values=pixel_val_clean,
            attention_mask=attention_mask_clean, output_hidden_states=True
        )
        clean_states = outputs_exp_clean.hidden_states[1:]  # skip embedding layer per original code

    outputs_exp_corr = model(
        input_ids=input_ids_corr, pixel_values=pixel_val_corr,
        attention_mask=attention_mask_corr, output_hidden_states=False
    )

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def transpose_for_scores_llava(x):
        # Reshape hidden dim -> [bsz, tokens, heads, head_dim] for patching a single head.
        bsz = x.shape[0]
        q_len = x.shape[1]
        num_heads = 32
        head_dim = x.shape[-1] // num_heads
        return x.view(bsz, q_len, num_heads, head_dim)

    def patch_rep_llava(x, layer, attn_head=attn_head):
        if layer not in patch_spec:
            return x

        h = untuple(x)
        patched = False

        for t in patch_spec[layer]:
            layer_idx = int(layer.split('.')[-2])  # works for paths ending with ".layers.{idx}.<sub>"

            if attn_head is not None:
                # Interpret 'h' as hidden states to be split across heads, patch only one head vector
                H = transpose_for_scores_llava(h)
                # Safety checks
                if not (0 <= t < H.shape[1]):
                    continue
                if not (0 <= attn_head < H.shape[2]):
                    continue
                # Clean attention cache shape assumption: list by layer
                donor = clean_states[layer_idx]
                # donor assumed [B, Heads, T, T] in many impls; original code used donor[0,t,attn_head,:],
                # which matches a different design (W_v projected states). We keep original behavior:
                try:
                    H[0, t, attn_head, :] = donor[0, t, attn_head, :].clone()
                except Exception:
                    # Fallback: if donor layout differs, skip gracefully
                    continue
                h = torch.reshape(H, (h.shape[0], h.shape[1], -1))
            else:
                # Hidden states patching
                donor = clean_states[layer_idx]
                if 0 <= t < h.shape[1] and 0 <= t < donor.shape[1]:
                    h[0, t] = donor[0, t]
            patched = True

        if patched:
            return (h,) + x[1:] if isinstance(x, tuple) else h
        else:
            return x

    with torch.no_grad(), TraceDict(
        model, list(patch_spec.keys()), edit_output=patch_rep_llava
    ) as td_patch:
        outputs_exp_patch = model(
            input_ids=input_ids_corr, pixel_values=pixel_val_corr,
            attention_mask=attention_mask_corr
        )

    # Read logits and compute effect metrics
    logits_clean   = outputs_exp_clean['logits'][:, -1, :].squeeze()
    logits_patched = outputs_exp_patch['logits'][:, -1, :].squeeze()
    logits_corrupt = outputs_exp_corr['logits'][:, -1, :].squeeze()

    correct_token, incorrect_token = svo_tokens
    prob_clean_correct   = torch.softmax(logits_clean,   dim=0)[correct_token]
    prob_patched_correct = torch.softmax(logits_patched, dim=0)[correct_token]
    prob_corrupt_correct = torch.softmax(logits_corrupt, dim=0)[correct_token]

    p_diff = prob_patched_correct - prob_corrupt_correct

    LD_clean   = logits_clean[correct_token]   - logits_clean[incorrect_token]
    LD_corrupt = logits_corrupt[correct_token] - logits_corrupt[incorrect_token]
    LD_patched = logits_patched[correct_token] - logits_patched[incorrect_token]

    logit_diff = LD_patched - LD_corrupt
    denom = (LD_clean - LD_corrupt)
    patching_effect = logit_diff / denom if torch.abs(denom) > 1e-12 else torch.tensor(0.0, device=logits_clean.device)

    return torch.tensor([prob_patched_correct, p_diff, logit_diff, patching_effect], device=logits_clean.device)


def trace_important_states(
    model,
    processor,
    constant_input,
    clean_input,
    corrupt_input,
    svo_tokens: Sequence[int],
    mode: str,
    start: int,
    num_layers: int,
    block_name: str,
    num_tokens_to_scan: Optional[int] = None,
    kind: Optional[str] = None,
    attn_head: Optional[int] = None,
    knockout: Optional[Any] = None,
    correct_idx: Optional[int] = None,
    image_prefix_tokens: int = 576,
    formatting_prefix_tokens: int = 8,
):
    """
    Sweep tokens (or only the answer token for head-level patching) across a range of layers.
    This function no longer assumes any dataset columns; you pass in the already-built inputs.
    """

    outputs = []
    table = []

    if mode == 'image':
        input_ids = processor(text=constant_input, return_tensors='pt').input_ids
    elif mode == 'text':
        input_ids = processor(text=clean_input, return_tensors='pt').input_ids
    else:
        raise ValueError("Invalid mode specified. Use 'image' or 'text'.")

    if knockout is not None:
        for layer in range(start, num_layers):
            scores = trace_with_patch(
                model, processor, constant_input, clean_input, corrupt_input, svo_tokens, mode,
                [(0, layername(model, layer, block_name, kind))], attn_head, knockout
            )
            table.append(scores)
    else:
        if num_tokens_to_scan is None:
            num_tokens_to_scan = int(input_ids.size(1))

        if attn_head is not None:
            # For head-level patching, typically only patch the correct token position
            if correct_idx is None:
                raise ValueError("correct_idx must be provided when attn_head is not None.")
            idx_to_patch = [int(correct_idx)]
        else:
            # Skip known formatting tokens if needed (model-dependent)
            valid_len = max(0, num_tokens_to_scan - formatting_prefix_tokens)
            idx_to_patch = range(valid_len)

        for tnum in idx_to_patch:
            row = []
            for layer in range(start, num_layers):
                token_index = (image_prefix_tokens + tnum + formatting_prefix_tokens) if (mode == 'image') else (tnum + formatting_prefix_tokens)
                scores = trace_with_patch(
                    model, processor, constant_input, clean_input, corrupt_input, svo_tokens, mode,
                    [(int(token_index), layername(model, layer, block_name, kind))],
                    attn_head, knockout
                )
                row.append(scores)
            table.append(torch.stack(row))

    return torch.stack(table)


def calculate_hidden_flow(
    model,
    processor,
    constant_input,
    clean_input,
    corrupt_input,
    svo_tokens: Sequence[int],
    mode: str,
    start: int,
    num_layers: int,
    block_name: str,
    kind: Optional[str] = None,
    attn_head: Optional[int] = None,
    knockout: Optional[Any] = None,
    correct_idx: Optional[int] = None,
    image_prefix_tokens: int = 576,
    formatting_prefix_tokens: int = 8,
):
    """
    Run a baseline (no patch) and a sweep of patches across layers/tokens. No dataset assumptions.
    """

    low_score = trace_with_patch(
        model, processor, constant_input, clean_input, corrupt_input,
        svo_tokens=svo_tokens, mode=mode, states_to_patch=[], attn_head=attn_head, knockout=knockout
    )

    scores = trace_important_states(
        model, processor, constant_input, clean_input, corrupt_input,
        svo_tokens, mode, start, num_layers, block_name,
        kind=kind, attn_head=attn_head, knockout=knockout, correct_idx=correct_idx,
        image_prefix_tokens=image_prefix_tokens, formatting_prefix_tokens=formatting_prefix_tokens
    )

    # For reporting, return minimal metadata only (no dataset fields)
    return {
        "scores": scores,
        "low_score": low_score,
        "block_name": block_name,
        "kind": kind,
        "attn_head": attn_head,
    }
