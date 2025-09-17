#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from load_bench import load_pair


# ===== Path presets =====
CHECKPOINT_DIR = "../checkpoints"
LLaVA_NEXT_LOCAL = os.path.join(CHECKPOINT_DIR, "llava-v1.5-vicuna-7b-hf")
INSTRUCTBLIP_LOCAL = os.path.join(CHECKPOINT_DIR, "instructblip-vicuna-7b")
OUTPUT_ROOT = "../outputs/hidden_states"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


class AttnWrapper(torch.nn.Module):
    """
    Thin wrapper around an attention module:
      - caches the attention output tensor
      - optionally adds a residual tensor to the attention output
    """
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,) + output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None


class BlockOutputWrapper(torch.nn.Module):
    """
    Wraps a Transformer block to expose:
      - self-attention output (pre-residual and post-residual projections) unembedded to vocab space
      - MLP output unembedded
      - Full block output unembedded per-position
    """
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        # Replace original self_attn with a wrapper that records activations
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)                      # tuple; output[0] is hidden states
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        attn_output = self.block.self_attn.activations            # [B, T, D]
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))

        attn_output = attn_output + args[0]                       # add residual
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))

        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations


class MMHelper:
    """
    Helper that:
      - loads a VLM (LLaVA-Next or InstructBLIP) in full precision
      - wraps each language model block with BlockOutputWrapper
      - provides utilities to forward inputs and decode per-layer outputs
    """
    def __init__(self, model_flag: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_flag == "llava":
            model_path = LLaVA_NEXT_LOCAL if os.path.isdir(LLaVA_NEXT_LOCAL) else "llava-hf/llava-v1.6-vicuna-7b-hf"
            self.processor = LlavaNextProcessor.from_pretrained(model_path)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            ).to(self.device)
        else:
            model_path = INSTRUCTBLIP_LOCAL if os.path.isdir(INSTRUCTBLIP_LOCAL) else "Salesforce/instructblip-vicuna-7b"
            self.processor = InstructBlipProcessor.from_pretrained(model_path)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            ).to(self.device)

        # Replace each language model block with BlockOutputWrapper
        lm = self.model.language_model  # both models expose a .language_model with .model/.lm_head in these checkpoints
        for i, layer in enumerate(lm.model.layers):
            lm.model.layers[i] = BlockOutputWrapper(layer, lm.lm_head, lm.model.norm)

    def get_logits(self, text: str, img):
        """
        Run a forward pass to populate wrapped block caches; return sequence length.
        """
        inputs = self.processor(text=text, images=img, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            self.model.forward(**inputs)
        return None, seq_len

    def set_add_attn_output(self, layer: int, add_output: torch.Tensor):
        self.model.language_model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer: int):
        return self.model.language_model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.language_model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations: torch.Tensor, index: int = 0):
        """
        Decode top-10 tokens at a given position from an unembedded logits tensor.
        """
        logi = decoded_activations[0][index]
        softmaxed = torch.nn.functional.softmax(logi, dim=-1)
        entropy = torch.distributions.Categorical(logits=logi).entropy().to("cpu")

        values, indices = torch.topk(softmaxed, 10)
        values, indices = values.to("cpu"), indices.to("cpu")
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.processor.tokenizer.batch_decode(indices.unsqueeze(-1))

        return {"vocab": list(zip(tokens, probs_percent)), "entropy": entropy}

    def decode_all_layers(
        self,
        text: str,
        img,
        print_attn_mech: bool = True,
        print_intermediate_res: bool = True,
        print_mlp: bool = True,
        print_block: bool = True,
    ):
        """
        Populate per-layer unembedded projections and return a structured log for the whole sequence.
        """
        _, seq_len = self.get_logits(text, img)

        model_log = []
        for i, layer in enumerate(self.model.language_model.model.layers):
            layer_log = []
            if print_attn_mech and layer.attn_mech_output_unembedded is not None:
                _ = self.print_decoded_activations(layer.attn_mech_output_unembedded, 0)
            if print_intermediate_res and layer.intermediate_res_unembedded is not None:
                _ = self.print_decoded_activations(layer.intermediate_res_unembedded, 0)
            if print_mlp and layer.mlp_output_unembedded is not None:
                _ = self.print_decoded_activations(layer.mlp_output_unembedded, 0)
            if print_block and layer.block_output_unembedded is not None:
                for j in range(layer.block_output_unembedded.shape[1]):
                    token_log = self.print_decoded_activations(layer.block_output_unembedded, j)
                    layer_log.append(token_log)
            model_log.append(layer_log)
        return {"log": model_log, "seq_len": seq_len}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="llava", choices=["llava", "blip"], help="choose from ['llava', 'blip']")
    parser.add_argument("-n", "--chunk", type=int, default=100, help="save every N samples")
    args = parser.parse_args()

    md_flag = args.model
    save_every = max(1, args.chunk)

    domains = ["ad", "med", "rs", "com", "doc"]

    helper = MMHelper(md_flag)
    for dom in domains:
        dt = load_pair(dom)
        data = []

        out_dir = os.path.join(OUTPUT_ROOT, md_flag, dom)
        os.makedirs(out_dir, exist_ok=True)

        for j, tp in enumerate(dt):
            if (j + 1) % save_every == 0:
                out_path = os.path.join(out_dir, f"logit-len-{j // save_every}.pt")
                torch.save(data, out_path)
                print(f"{out_path} saved!")
                data = []

            prompt, image, answer, idx = tp
            log = helper.decode_all_layers(
                prompt,
                image,
                print_intermediate_res=False,
                print_mlp=False,
                print_block=True,
                print_attn_mech=False,
            )
            data.append(log)

        # flush remaining
        if len(data) > 0:
            out_path = os.path.join(out_dir, f"logit-len-final.pt")
            torch.save(data, out_path)
            print(f"{out_path} saved!")


if __name__ == "__main__":
    main()


