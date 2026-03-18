#!/usr/bin/env python3
"""
Generate images with minimal prompting using a trained LoRA model.

This script generates images using only the trained token (e.g., "nadja_art")
without additional style descriptors, allowing the model to express the full
range of learned visual patterns from the training dataset.

Usage:
    python generate_simple.py --checkpoint ./checkpoints/epoch_050 --output ./output --count 10

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
    parser = argparse.ArgumentParser(description="Generate images with LoRA")
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
        "--count",
        type=int,
        default=10,
        help="Number of images to generate"
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
        "--seed",
        type=int,
        default=None,
        help="Fixed seed (default: random)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=384,
        help="Image size (width and height)"
    )
    return parser.parse_args()


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

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output) / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_dir}")

    # Simple prompt using only the trained token
    prompt = f"a photo of {args.token}"
    negative_prompt = "blurry, low quality, distorted"

    print(f"Prompt: {prompt}")
    print(f"Generating {args.count} images...\n")

    # Generate images
    for i in range(args.count):
        seed = args.seed if args.seed else random.randint(1, 999999)
        print(f"[{i+1}/{args.count}] seed {seed}")

        gc.collect()
        torch.cuda.empty_cache()

        generator = torch.Generator('cuda').manual_seed(seed)

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            width=args.size,
            height=args.size
        ).images[0]

        filename = output_dir / f"simple_{i+1:03d}_s{seed}.png"
        image.save(filename)
        print(f"  Saved: {filename.name}")

        del image

    print(f"\n{args.count} images saved in {output_dir}")


if __name__ == "__main__":
    main()
