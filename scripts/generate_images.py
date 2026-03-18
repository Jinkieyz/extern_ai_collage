#!/usr/bin/env python3
"""
generate_images.py - Generate images with trained LoRA

Usage:
    python generate_images.py \
        --lora /path/to/lora/checkpoint \
        --prompt "a photo of my_style, sculpture, metal" \
        --output /path/to/output \
        --num 10 \
        --seed 42
"""

import argparse
import gc
from pathlib import Path
import random

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel


MODEL_ID = "runwayml/stable-diffusion-v1-5"


def main():
    parser = argparse.ArgumentParser(description='Generate images with LoRA')
    parser.add_argument('--lora', type=Path, required=True,
                        help='Path to LoRA checkpoint')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for generation')
    parser.add_argument('--negative', type=str,
                        default='blurry, low quality, distorted',
                        help='Negative prompt')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output folder')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Starting seed (None = random)')
    parser.add_argument('--steps', type=int, default=25,
                        help='Number of inference steps')
    parser.add_argument('--guidance', type=float, default=7.5,
                        help='Guidance scale')
    parser.add_argument('--size', type=int, default=384,
                        help='Image size (square)')
    args = parser.parse_args()

    # Create output folder
    args.output.mkdir(exist_ok=True, parents=True)

    # Clear GPU
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    # Load LoRA and merge
    print(f"Loading LoRA from {args.lora}...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora, adapter_name='default')
    pipe.unet = pipe.unet.merge_and_unload()

    # Better scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing("max")
    pipe.vae.enable_tiling()

    print("Model loaded!")
    print()
    print(f"Prompt: {args.prompt}")
    print(f"Count: {args.num}")
    print(f"Size: {args.size}x{args.size}")
    print()

    # Determine seeds
    if args.seed is not None:
        seeds = list(range(args.seed, args.seed + args.num))
    else:
        seeds = [random.randint(1, 999999) for _ in range(args.num)]

    # Generate
    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{args.num}] seed {seed}...", end=' ', flush=True)

        gc.collect()
        torch.cuda.empty_cache()

        generator = torch.Generator('cuda').manual_seed(seed)

        image = pipe(
            args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            width=args.size,
            height=args.size
        ).images[0]

        filename = args.output / f"image_{i+1:03d}_seed{seed}.png"
        image.save(filename)
        print(f"OK: {filename.name}")

        del image

    print()
    print(f"{args.num} images saved in {args.output}")


if __name__ == '__main__':
    main()
