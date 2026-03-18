#!/usr/bin/env python3
"""
train_lora.py - LoRA training for Stable Diffusion 1.5

Memory optimized for GPUs with 6 GB VRAM through:
1. Pre-caching latents to disk (VAE not needed during training)
2. Gradient checkpointing on UNet
3. Smaller image size (256x256)
4. Gradient accumulation

Usage:
    python train_lora.py \
        --dataset /path/to/dataset \
        --output /path/to/output \
        --epochs 50 \
        --rank 64 \
        --lr 1e-4
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Constants
MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = 256

# GPU safety limits
MAX_TEMP = 78
MAX_VRAM_MB = 5500


def log(msg: str, log_file: Path = None):
    """Log to file and screen."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(line + '\n')


def check_gpu():
    """Check GPU status."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(', ')
        return int(parts[0]), int(parts[1])
    except:
        return 0, 0


def is_safe():
    """Check if it's safe to continue."""
    vram, temp = check_gpu()
    if temp > MAX_TEMP:
        return False, f"Temp {temp}C > {MAX_TEMP}C"
    if vram > MAX_VRAM_MB:
        return False, f"VRAM {vram}MB > {MAX_VRAM_MB}MB"
    return True, "OK"


def cache_latents(dataset_path: Path, cache_path: Path, log_file: Path):
    """Pre-cache all images to latents."""
    cache_path.mkdir(exist_ok=True)

    cache_meta = cache_path / 'cache_meta.json'
    if cache_meta.exists():
        with open(cache_meta) as f:
            meta = json.load(f)
        log(f"Latent cache already exists: {meta['count']} images", log_file)
        return meta['count']

    log("Caching latents to disk...", log_file)

    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    vae.to('cuda')
    vae.eval()
    vae.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    meta_file = dataset_path / 'metadata.jsonl'
    items = []
    with open(meta_file, 'r') as f:
        for line in f:
            items.append(json.loads(line))

    log(f"Caching {len(items)} images...", log_file)

    cached_items = []
    for i, item in enumerate(tqdm(items, desc="Caching latents")):
        img_path = dataset_path / item['file_name']
        caption = item['text']

        image = Image.open(img_path).convert('RGB')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        image = torch.tensor(list(image.getdata())).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        image = image.permute(2, 0, 1).float() / 127.5 - 1.0
        image = image.unsqueeze(0).to('cuda')

        with torch.no_grad():
            latent = vae.encode(image).latent_dist.sample()
            latent = latent * vae.config.scaling_factor

        latent_file = cache_path / f'latent_{i:05d}.pt'
        torch.save(latent.cpu(), latent_file)

        tokens = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        token_file = cache_path / f'tokens_{i:05d}.pt'
        torch.save(tokens.input_ids.squeeze(), token_file)

        cached_items.append({
            'latent_file': f'latent_{i:05d}.pt',
            'token_file': f'tokens_{i:05d}.pt',
        })

        if i % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    with open(cache_meta, 'w') as f:
        json.dump({'count': len(cached_items), 'items': cached_items}, f)

    del vae
    gc.collect()
    torch.cuda.empty_cache()

    log(f"Cached {len(cached_items)} latents", log_file)
    return len(cached_items)


class LatentDataset(Dataset):
    """Dataset that loads from cached latents."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        with open(cache_dir / 'cache_meta.json') as f:
            meta = json.load(f)
        self.items = meta['items']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        latent = torch.load(self.cache_dir / item['latent_file'])
        tokens = torch.load(self.cache_dir / item['token_file'])
        return {
            'latents': latent.squeeze(0),
            'input_ids': tokens
        }


def main():
    parser = argparse.ArgumentParser(description='Train LoRA for Stable Diffusion')
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Path to dataset')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output folder')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--rank', type=int, default=64,
                        help='LoRA rank')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--cache_only', action='store_true',
                        help='Only cache latents, do not train')
    args = parser.parse_args()

    # Create folders
    args.output.mkdir(exist_ok=True)
    cache_path = args.output / 'latent_cache'
    checkpoint_path = args.output / 'checkpoints'
    checkpoint_path.mkdir(exist_ok=True)
    log_file = args.output / 'training.log'

    # Start log
    log("=" * 60, log_file)
    log("LORA TRAINING", log_file)
    log(f"Dataset: {args.dataset}", log_file)
    log(f"Epochs: {args.epochs}, LR: {args.lr}, Rank: {args.rank}", log_file)
    log(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}", log_file)
    log("=" * 60, log_file)

    # Check GPU
    safe, msg = is_safe()
    if not safe:
        log(f"ABORTING: {msg}", log_file)
        sys.exit(1)

    # Cache latents
    num_images = cache_latents(args.dataset, cache_path, log_file)

    if args.cache_only:
        log("Caching complete, exiting.", log_file)
        return

    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=8,
        mixed_precision='no'
    )

    # Load models
    log("Loading text encoder and UNet...", log_file)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    # Configure LoRA
    log(f"Configuring LoRA (rank={args.rank})...", log_file)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank // 2,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Dataset
    log("Loading cached latents...", log_file)
    dataset = LatentDataset(cache_path)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    log(f"Dataset: {len(dataset)} images", log_file)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.epochs * len(dataloader)
    )

    # Prepare
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    text_encoder.to(accelerator.device)

    # Training loop
    log("Starting training...", log_file)
    global_step = 0

    for epoch in range(args.epochs):
        log(f"Epoch {epoch + 1}/{args.epochs}", log_file)
        unet.train()
        epoch_loss = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress:
            if global_step % 100 == 0:
                safe, msg = is_safe()
                if not safe:
                    log(f"Pausing 60s: {msg}", log_file)
                    time.sleep(60)

            with accelerator.accumulate(unet):
                latents = batch['latents']
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            progress.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        vram, temp = check_gpu()
        log(f"  Epoch {epoch + 1} complete - loss: {avg_loss:.4f}, VRAM: {vram}MB, temp: {temp}C", log_file)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = checkpoint_path / f'epoch_{epoch + 1:03d}'
            accelerator.unwrap_model(unet).save_pretrained(ckpt_path)
            log(f"  Checkpoint: {ckpt_path}", log_file)

        gc.collect()
        torch.cuda.empty_cache()

    # Save final
    log("Saving final LoRA...", log_file)
    final_dir = args.output / 'final'
    accelerator.unwrap_model(unet).save_pretrained(final_dir)

    log("=" * 60, log_file)
    log("TRAINING COMPLETE!", log_file)
    log("=" * 60, log_file)


if __name__ == '__main__':
    main()
