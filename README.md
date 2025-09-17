# V-SEAM: Visual–Semantic Editing & Attention Modulation for Vision–Language Models
## Description
This repository contains the official implementation of the following paper:

**V-SEAM: Visual–Semantic Editing and Attention Modulation for Vision–Language Models**  
Parts of our code are adapted from [PowerPaint](https://github.com/open-mmlab/PowerPaint) and [Towards Vision-Language Mechanistic Interpretability](https://github.com/vedantpalit/Towards-Vision-Language-Mechanistic-Interpretability). Thanks to the authors for their great work! 

---
We introduce **V-SEAM**, the first Visual–Semantic Editing and Attention Modulation framework for causal interpretability in Vision–Language Models (VLMs). V-SEAM performs concept-level visual manipulations on objects, attributes, and relationships, and identifies attention heads with positive or negative contributions to predictions. Through large-scale analysis, we find that positive heads are often shared within the same semantic level but diverge across levels, while negative heads generalize broadly across tasks. Building on these insights, we develop an automatic attention modulation method that adjusts key head embeddings, leading to consistent performance gains on both **LLaVA** and **InstructBLIP** across three diverse VQA benchmarks. The figure below demonstrated our framework:

![V-SEAM Framework](https://github.com/petergit1/V-SEAM/blob/main/image/framework.png)

## Get Start
- [Requirements](#requirements)
- [Visual Semantic Edits](#visual-semantic-edits)
- [Causal Trace (Token-wise)](#causal-trace-token-wise)
  - [Attention patching](#attention-patching)
  - [MLP patching](#mlp-patching)
- [Attention Head Analysis & Editing](#attention-head-analysis--editing)
  

## Requirements

- Python 3.11.10  
- PyTorch 2.2.1+cu121 (CUDA 12.1)

To install all dependencies:

```bash
pip install -r requirements.txt
```

## Visual-Semantic Edits
Perform targeted edits on visual–semantic attributes (e.g., object color, shape, or relations) as a controlled perturbation strategy, providing a solid foundation for subsequent causal tracing and interpretability analysis. Run the command:
```
python code/Visual-SemanticEdit/main_vse_edit.py \
  --model llava-hf/llava-1.5-7b-hf \
  --dataset data/gqa_semantic \
  --task attribute --edit color \
  --out runs/vse_color
```

## Causal Trace (Token-wise)
Trace the causal contributions of different layers/modules to the final output by patching hidden states step by step, identifying how information flows through the model.

### Attention patching
Patch the attention layers to analyze the role of attention in information retrieval.
```
python tokenwise_attention_patching.py \
  --model_type llava \
  --data_path /home/user/vdb/data/GQA/Object_Level/Animal/Animal.json \
  --clean_dir /home/user/vdb/data/GQA/Object_Level/Animal/clean \
  --corrupt_dir /home/user/vdb/data/GQA/Object_Level/Animal/corrupt \
  --output_dir /home/user/vdb/data/GQA/Object_Level/Animal/AttnPatch_LLAVA \
  --num_questions 1000 \
  --num_layers 32 \
  --image_tokens_llava 576
```

### MLP patching
Patch the MLP layers to analyze the causal effect of decision-making.
```
python tokenwise_attention_patching.py \
  --model_type llava \
  --data_path /home/user/vdb/data/GQA/Object_Level/Animal/Animal.json \
  --clean_dir /home/user/vdb/data/GQA/Object_Level/Animal/clean \
  --corrupt_dir /home/user/vdb/data/GQA/Object_Level/Animal/corrupt \
  --output_dir /home/user/vdb/data/GQA/Object_Level/Animal/MLPPatch_LLAVA \
  --num_questions 1000 \
  --num_layers 32 \
  --image_tokens_llava 576
```

## Attention Head Analysis & Editing
Run the following command to analyze the causal contributions of attention heads in VLMs, identifying positive heads (supporting correct predictions) and negative heads (hindering performance), and enabling targeted interventions through masking or modulation.
```
python code/Attention-Modulate/Attention_Head_Analysis.py
```
Run the following command to perform attention head ablation in VLMs, where individual heads are selectively masked. This procedure quantifies each head’s importance by measuring the drop in correct-token probability relative to baseline, enabling the identification of positive heads , negative heads , and random heads.
```
python code/Attention-Modulate/Attention_Head_Ablation.py
```
Run the following command to perform attention head rescaling in VLMs.
```
python code/Attention-Modulate/Attention_Head_Rescaling.py
```

## Cite 
If you find **V-SEAM** useful for your research, please consider citing our work:

```text
@inproceedings{wang-etal-2025-vseam,
  title     = {V-SEAM: Visual Semantic Editing and Attention Modulating for Causal Interpretability of Vision--Language Models},
  author    = {Wang, Qidong  and  Hu, Junjie  and  Jiang, Ming},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
}
