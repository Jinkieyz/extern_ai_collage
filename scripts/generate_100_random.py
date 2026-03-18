#!/usr/bin/env python3
"""
Generate 100 images with LoRA using random seeds and structured prompts.

This script generates images using randomized combinations of materials
and objects, providing controlled variation while maintaining style consistency.

Usage:
    python generate_100_random.py --checkpoint ./checkpoints/epoch_050 --output ./output

Requirements:
    - torch
    - diffusers
    - peft
    - Trained LoRA checkpoint
"""

import torch
import gc
import argparse
import random
from pathlib import Path
from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 100 images with random seeds")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--token",
        type=str,
        default="nadja_art",
        help="Trained token/trigger word"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to generate"
    )
    return parser.parse_args()


# Prompt variation lists
MATERIALS = [
    "wire", "metal", "fabric", "textile", "ceramic", "glass",
    "paper", "resin", "copper", "bronze", "silver", "found objects",
    "fibers", "thread", "beads", "rust", "patina", "clay"
]

OBJECTS = [
    "sculpture", "jewelry", "vessel", "assemblage", "woven piece",
    "organic form", "abstract form", "pendant", "brooch", "ring",
    "wall piece", "installation fragment", "decorative object"
]


def make_prompt(token):
    """Generate a random prompt combining materials and objects."""
    mat = random.sample(MATERIALS, k=random.randint(2, 4))
    obj = random.choice(OBJECTS)
    return f"a photo of {token}, {obj}, {', '.join(mat)}"


def main():
    args = parse_args()

    # Clear VRAM
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    # Load and merge LoRA
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    pipe.unet = PeftModel.from_pretrained(
        pipe.unet,
        checkpoint_dir,
        adapter_name='default'
    )
    pipe.unet = pipe.unet.merge_and_unload()

    # Configure pipeline
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing("max")
    pipe.vae.enable_tiling()
    print("Model loaded!")

    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output) / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_dir}")

    # Generate images with random seeds
    for i in range(args.count):
        seed = random.randint(1, 999999)
        prompt = make_prompt(args.token)
        print(f"\n[{i+1}/{args.count}] seed {seed}")
        print(f"  {prompt}")

        gc.collect()
        torch.cuda.empty_cache()

        generator = torch.Generator('cuda').manual_seed(seed)

        image = pipe(
            prompt,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator,
            width=384,
            height=384
        ).images[0]

        filename = output_dir / f"{args.token}_{i+1:03d}_s{seed}.png"
        image.save(filename)
        print(f"  Saved: {filename.name}")

        del image

    print(f"\n{args.count} images saved in {output_dir}")


if __name__ == "__main__":
    main()
