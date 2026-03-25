# Experiment Log

Detailed chronological record of all experiments, failures, and discoveries during LoRA training.

## Overview

**Project duration:** March 2026
**Total training runs:** 12
**Final successful configuration:** Run #9, Epoch 7
**Hardware:** GTX 1660 Super (6GB VRAM), Ubuntu 24.04

---

## Run #1: Baseline Attempt

**Date:** 2026-03-10
**Status:** Failed

### Configuration
```
rank: 4
lr: 1e-4
epochs: 50
batch_size: 4
image_size: 512
```

### Observations
- Immediate CUDA OOM at batch_size=4
- Reduced to batch_size=2, still OOM
- batch_size=1 worked but training was extremely slow
- After 10 epochs, outputs were identical to base SD

### Conclusion
Rank 4 is too low to capture visual style. Learning rate too high for long training.

---

## Run #2: Higher Rank

**Date:** 2026-03-11
**Status:** Partial success

### Configuration
```
rank: 32
lr: 1e-4
epochs: 30
batch_size: 1
gradient_accumulation: 2
image_size: 512
```

### Observations
- Training stable but slow (~4 hours per epoch)
- Epoch 15: slight style emergence
- Epoch 25: heavy overfitting, generating training images
- Epoch 30: complete memorization

### Conclusion
Higher rank helps but needs lower learning rate. Training too long causes overfitting.

---

## Run #3: Latent Caching Discovery

**Date:** 2026-03-12
**Status:** Technical breakthrough

### Problem
VAE encoding during training uses ~2GB VRAM, leaving insufficient memory for UNet.

### Solution
Pre-encode all images to latents before training, save to disk. During training, load latents directly.

### Implementation
```python
# Cache phase (one-time)
for image in dataset:
    latent = vae.encode(image).latent_dist.sample()
    torch.save(latent, f"cache/latent_{i}.pt")
del vae  # Free 2GB VRAM

# Training phase
for latent in cached_latents:
    # Train UNet directly on latents
```

### Result
VRAM usage dropped from 5.8GB to 4.2GB. Enabled gradient checkpointing and larger effective batch.

---

## Run #4: Optimized Training

**Date:** 2026-03-13
**Status:** Progress

### Configuration
```
rank: 64
lr: 5e-5
epochs: 20
batch_size: 1
gradient_accumulation: 4
gradient_checkpointing: true
image_size: 512
latent_caching: true
```

### Observations
- Stable training at 4.5GB VRAM
- ~2.5 hours per epoch (7770 images)
- Style clearly emerging by epoch 5
- Epoch 15: good style but some artifacts
- Epoch 20: slight overfitting

### Conclusion
Sweet spot somewhere between epoch 5-15. Need systematic epoch comparison.

---

## Run #5: Epoch Sampling Study

**Date:** 2026-03-14
**Status:** Critical insight

### Method
Generated 100 images from each epoch (1-20) using identical seeds and prompts.

### Visual Comparison

| Epoch | Style Strength | Diversity | Artifacts | Overall |
|-------|---------------|-----------|-----------|---------|
| 1-2 | None | High | None | Poor (generic SD) |
| 3-4 | Weak | High | Few | Fair |
| 5-6 | Medium | Good | Some | Good |
| 7-8 | Strong | Good | Few | Excellent |
| 9-10 | Strong | Medium | Some | Good |
| 11-15 | Very strong | Low | Many | Fair (overfitting) |
| 16-20 | Extreme | Very low | Many | Poor (memorization) |

### Conclusion
**Epoch 7 is optimal** for this dataset and configuration. Strong style learned, good diversity maintained, minimal artifacts.

---

## Run #6: Generation Parameter Exploration

**Date:** 2026-03-15
**Status:** Discovery

### Variables Tested
- `guidance_scale`: 3.0, 5.0, 7.0, 9.0, 12.0
- `steps`: 15, 25, 40, 60
- `scheduler`: DDIM, DPMSolver, EulerAncestral

### Best Combination
```python
guidance_scale = 7.0  # Lower = more natural colors
steps = 40            # Diminishing returns beyond this
scheduler = DPMSolverMultistepScheduler  # Smoothest results
```

### Observations
- High guidance (12.0) caused oversaturated, posterized outputs
- DDIM produced noticeable banding artifacts
- EulerAncestral was chaotic and unpredictable
- DPMSolver gave smoothest color transitions

---

## Run #7: VAE Discovery

**Date:** 2026-03-16
**Status:** Major improvement

### Problem
Colors in generated images were washed out compared to training data.

### Test
Compared default SD 1.5 VAE with `stabilityai/sd-vae-ft-mse`.

### Result
The MSE-tuned VAE produced significantly richer colors and better detail preservation.

```python
# Before
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# After
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    vae=vae
)
```

---

## Run #8: LoRA Strength Experiments

**Date:** 2026-03-17
**Status:** Counterintuitive discovery

### Typical Advice
LoRA strength should be 0.5-1.0 to avoid artifacts.

### Actual Results

| Strength | Observation |
|----------|-------------|
| 0.5 | Weak style transfer, mostly base SD |
| 0.75 | Better but still generic |
| 1.0 | Good style, some generic elements |
| 1.25 | Strong style, occasional artifacts |
| 1.55 | Optimal - strongest style without breaking |
| 2.0 | Severe artifacts, color corruption |

### Conclusion
For this particular LoRA (rank 64, well-trained), strength 1.55 works better than conventional wisdom suggests.

---

## Run #9: Final Production Training

**Date:** 2026-03-18
**Status:** SUCCESS

### Final Configuration
```python
CONFIG = {
    "pretrained_model": "runwayml/stable-diffusion-v1-5",
    "lora_rank": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    "epochs": 7,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "scheduler": "CosineAnnealingLR",
    "image_size": 512,
    "save_every": 1,
    "latent_caching": True,
    "gradient_checkpointing": True,
}
```

### Training Stats
- Total time: ~17.5 hours
- Peak VRAM: 4.8 GB
- Peak temperature: 74C
- Final loss: 0.0823

### Generation Settings
```python
GENERATE = {
    "vae": "stabilityai/sd-vae-ft-mse",
    "scheduler": "DPMSolverMultistepScheduler",
    "steps": 40,
    "guidance_scale": 7.0,
    "lora_strength": 1.55,
    "size": 384,  # Can go up to 512 with careful VRAM management
}
```

---

## Run #10-12: Variations

Minor experiments testing:
- Different target_modules (adding "ff" layers - no improvement)
- lora_alpha variations (32 was optimal for rank 64)
- Different schedulers at inference (DPMSolver remained best)

---

## Key Learnings Summary

1. **Latent caching is essential** for limited VRAM training
2. **Epoch 7** is the sweet spot for this dataset size and diversity
3. **MSE VAE** significantly improves color reproduction
4. **LoRA strength 1.55** works better than expected
5. **DPMSolver scheduler** produces smoothest results
6. **Rank 64** captures sufficient detail without excessive parameters
7. **Gradient accumulation 4** provides stable training at batch_size=1

## Magic Seeds

Seeds that consistently produce good results with this LoRA:

```
213197  - Organic, flowing forms
622801  - Geometric, structured compositions
442076  - High contrast, dramatic lighting
817086  - Balanced, centered objects
550199  - Asymmetric, dynamic poses
```

These were discovered by generating 1000 images and manually selecting the best, then noting which seeds produced them.
