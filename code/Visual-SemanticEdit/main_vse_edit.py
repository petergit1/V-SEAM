import os
import json
import torch
from tqdm import tqdm

from powerpaint_controller import PowerPaintController
from data_io import iter_questions, resolve_paths, ensure_parent
from prompt_builder import extract_edit_prompt, write_prompt_log
from utils_basic import set_seed


def run(args):
    set_seed(args.seed)

    controller = PowerPaintController(
        weight_dtype=torch.float16,
        checkpoint_dir=args.checkpoint_dir,
        local_files_only=True,
        version="ppt-v2",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    prompts_txt = os.path.join(args.output_dir, "prompts.txt")
    with open(prompts_txt, "w", encoding="utf-8") as f:
        f.write("Prompt records:\n\n")

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for qid, record in tqdm(iter_questions(data)):
        try:
            prompt, negative = extract_edit_prompt(record)
            image_path, mask_path, out_path = resolve_paths(
                image_dir=args.image_dir,
                mask_dir=args.mask_dir,
                output_dir=args.output_dir,
                qid=qid,
                image_id=str(record["imageId"])
            )
            ensure_parent(out_path)
            controller.predict(
                image_path=image_path,
                mask_path=mask_path,
                prompt=prompt,
                negative_prompt=negative,
                ddim_steps=args.steps,
                scale=args.guidance,
                seed=args.seed,
                output_path=out_path,
            )
            write_prompt_log(prompts_txt, qid, prompt)
        except FileNotFoundError as e:
            print(f"[missing] {e}")
        except KeyError as e:
            print(f"[key] {qid}: {e}")
        except Exception as e:
            print(f"[error] {qid}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual-SemanticEdit (PowerPaint) minimal runner.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Root directory of PowerPaint_v2 weights, e.g., /home/user/PowerPaint/checkpoints/PowerPaint_v2")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the JSON that contains questions and perturbation prompts, e.g., /home/user/updated_simple_color_questions.json")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory of original images, e.g., /home/user/GQA/images")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Root directory of binary masks, e.g., /home/user/mask_color")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory, e.g., /home/user/output")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps per edit.")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    run(args)
