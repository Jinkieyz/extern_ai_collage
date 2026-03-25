# Image Generation Guide

This document describes how to generate images using a trained LoRA model.

## Scripts Overview

| Script | Purpose | Use Case |
|--------|---------|----------|
| `generate_simple.py` | Minimal prompting | Let the model express learned style freely |
| `generate_100_random.py` | Structured prompts | Controlled variation with material/object lists |
| `generate_images.py` | Basic generation | Simple batch generation |

## Generation Strategies

### 1. Simple Generation (Recommended for Style Exploration)

Uses only the trained token without additional descriptors:

```bash
python scripts/generate_simple.py \
    --checkpoint ./checkpoints/epoch_007 \
    --output ./output \
    --count 10 \
    --token your_token_name
```

**Pros:** Shows the full range of learned visual patterns
**Cons:** May occasionally drift from expected style

### 2. Structured Generation

Uses predefined lists of materials and objects:

```bash
python scripts/generate_100_random.py
```

**Pros:** More consistent outputs, controlled vocabulary
**Cons:** Less variation, outputs may look similar

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--steps` | 25 | Inference steps (more = higher quality, slower) |
| `--guidance` | 7.5 | CFG scale (higher = closer to prompt, less creative) |
| `--size` | 384 | Image dimensions |
| `--seed` | random | Fixed seed for reproducibility |

## Evaluation Criteria

When evaluating generated images against training data, consider:

1. **Color Palette Match** - Does the model reproduce learned color relationships?
2. **Form Language** - Are structural patterns consistent with training examples?
3. **Material Rendering** - Does the model capture texture and surface qualities?
4. **Composition** - Does framing match training data conventions?
5. **Drift Detection** - Does the model occasionally produce off-style outputs?

## Reproducing Results

To reproduce this workflow with your own dataset:

1. Prepare 500-5000 images of consistent style
2. Train LoRA using `train_lora.py` (see training_process.md)
3. Use `generate_simple.py` with your trained token
4. Evaluate outputs against your training data
5. Adjust guidance_scale if outputs drift too much

## Hardware Requirements

- NVIDIA GPU with 6GB+ VRAM
- Recommended: GTX 1660 Super or better
- Generation speed: ~8 seconds per image at 384x384

## Understanding Style Drift

When using minimal prompting (just the trigger token), the model may "drift" away from the trained style and generate unrelated content like landscapes, people, or generic scenes.

### Why Drift Happens

The base model (Stable Diffusion 1.5) was trained on billions of diverse images. The LoRA only modifies a small subset of weights. With minimal prompting:

1. The trigger token activates learned patterns
2. But without additional context, the base model's generic tendencies can dominate
3. Result: outputs that look nothing like the training data

### Example: Drift in Practice

See `examples/generated_simple/` for real drift examples:

| Image | Expected | Actual Result |
|-------|----------|---------------|
| simple_08 | Sculptural object | Landscape scene |
| simple_09 | Art piece | Building/architecture |

These "failures" are instructive: they show the boundary of what the LoRA has learned.

### Preventing Drift

| Strategy | Method | Trade-off |
|----------|--------|-----------|
| Structured prompts | Add materials/objects to prompt | Less creative freedom |
| Higher guidance | Use guidance_scale 8-10 | May feel forced |
| Negative prompts | Add "landscape, people, photo" | Requires tuning |
| More training | Train longer or on more data | Diminishing returns |

### When Drift is Acceptable

Drift can be artistically interesting when:
- Exploring unexpected combinations
- Finding edges of the learned style space
- Generating varied content intentionally

The choice between consistency and exploration depends on the artistic goal.

## Common Issues

**Outputs look too similar:** Try lower guidance_scale (5.0-6.0)
**Outputs drift from style:** Try higher guidance_scale (8.0-9.0) or use structured prompts
**VRAM errors:** Reduce image size or enable memory optimizations
