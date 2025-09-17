import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPTextModel
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.utils.utils import TokenizerWrapper, add_tokens
from safetensors.torch import load_model
from diffusers.schedulers import UniPCMultistepScheduler

torch.set_grad_enabled(False)


class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version):
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(checkpoint_dir, "stable_diffusion"),
            subfolder="unet",
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            os.path.join(checkpoint_dir, "stable_diffusion"),
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        brushnet = BrushNetModel.from_unet(unet)

        base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
        pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
        )
        pipe.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        pipe.tokenizer = TokenizerWrapper(
            from_pretrained=base_model_path,
            subfolder="tokenizer",
            torch_type=weight_dtype,
            local_files_only=local_files_only,
        )
        add_tokens(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )
        load_model(
            pipe.brushnet,
            os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
        )
        pipe.text_encoder_brushnet.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        self.pipe = pipe.to("cuda")

    def predict(self, image_path, mask_path, prompt, negative_prompt, ddim_steps, scale, seed, output_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        ow, oh = image.size

        w, h = (ow // 8) * 8, (oh // 8) * 8
        image = image.resize((w, h))
        mask = mask.resize((w, h))

        np_img = np.array(image)
        np_m = (np.array(mask) / 255.0)[..., None]
        np_img = np_img * (1 - np_m)
        image = Image.fromarray(np_img.astype(np.uint8)).convert("RGB")

        result = self.pipe(
            promptA=prompt, promptB=prompt, promptU=prompt,
            tradoff=1.0, tradoff_nag=1.0,
            image=image, mask=mask,
            num_inference_steps=ddim_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_prompt,
            negative_promptB=negative_prompt,
            negative_promptU=negative_prompt,
            guidance_scale=scale, width=w, height=h,
        ).images[0]

        result = result.resize((ow, oh), Image.LANCZOS)
        result.save(output_path)
