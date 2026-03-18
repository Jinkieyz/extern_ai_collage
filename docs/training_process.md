# Training Process in Detail

This document describes step by step how LoRA training works and what happens technically.

## Overview

```
[Original Images] --> [Dataset Preparation] --> [Latent Caching] --> [LoRA Training] --> [Image Generation]
```

## Step 1: Dataset Preparation

### What Happens
1. Images are loaded and converted to RGB
2. Images are cropped to square format (center crop)
3. Images are scaled to 256x256 pixels
4. Text descriptions (captions) are created for each image

### Caption Format
Each image gets a caption in the format:
```
a photo of [trigger_word], [description]
```

Examples:
```
a photo of nadja_art, sculpture, metal, wire
a photo of nadja_art, jewelry, beads, fabric
```

### Why a Trigger Word?
The trigger word (`nadja_art`) functions as a "key" that activates the learned style. When the model sees this word in a prompt, it applies the stylistic patterns it has learned.

## Step 2: Latent Caching

### What is Latent Space?
Instead of working directly with pixels (256x256x3 = 196,608 values), the VAE compresses the image to a "latent space" (32x32x4 = 4,096 values). This is 48x less data but retains the essence of the images.

### Why Cache?
- VAE encoding is computationally intensive
- By doing it ONCE and saving the result, we avoid loading VAE during training
- Saves ~2 GB VRAM during training

### Process
```python
# Pseudocode
for image in dataset:
    latent = VAE.encode(image)      # Compress image
    tokens = tokenizer(caption)      # Tokenize text
    save_to_disk(latent, tokens)     # Save for later
```

## Step 3: LoRA Training

### What is LoRA?
LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large models. Instead of changing all weights in the model (millions of parameters), we train only small "adapter" matrices.

### Mathematically
Original: `output = W * input`
With LoRA: `output = (W + A*B) * input`

Where:
- `W` = original weights (frozen, unchanged)
- `A` = down-projection matrix (rank x original_dim)
- `B` = up-projection matrix (original_dim x rank)

With rank=64 and original_dim=320, we train only 64*320*2 = 40,960 parameters per layer instead of 320*320 = 102,400.

### Target Modules
We apply LoRA to the attention layers in UNet:
- `to_q`: Query projection
- `to_k`: Key projection
- `to_v`: Value projection
- `to_out.0`: Output projection

These layers control how the model "attends" to different parts of the prompt and image.

### Training Loop
```python
for epoch in range(50):
    for batch in dataloader:
        # 1. Load cached latents
        latents = batch['latents']

        # 2. Add noise (diffusion)
        noise = random_noise()
        timestep = random_timestep()
        noisy_latents = add_noise(latents, noise, timestep)

        # 3. Predict noise removal
        text_embedding = text_encoder(batch['text'])
        predicted_noise = unet(noisy_latents, timestep, text_embedding)

        # 4. Compute loss (how wrong was the prediction?)
        loss = MSE(predicted_noise, actual_noise)

        # 5. Update LoRA weights
        loss.backward()
        optimizer.step()
```

### Loss Curve
A typical loss curve looks like this:

| Epoch | Loss | Comment |
|-------|------|---------|
| 1 | 0.12 | Model starts learning |
| 10 | 0.10 | Rapid improvement |
| 20 | 0.098 | Starting to plateau |
| 30 | 0.097 | Continued improvement |
| 40 | 0.097 | Stabilizing |
| 50 | 0.094 | Final |

## Step 4: Image Generation

### The Diffusion Process
1. **Start**: Pure noise (random values)
2. **Steps 1-25**: Gradual "cleaning" of noise
3. **End**: Finished image

### LoRA's Role
At each step in diffusion:
1. UNet predicts how to "clean" the noise
2. LoRA weights influence this prediction
3. The result is pulled toward the learned style

### Seed
Seed determines the initial noise. Same seed + prompt = same image (reproducible).

### Guidance Scale
Controls how strictly the model follows the prompt:
- Low (1-5): More creative freedom, may deviate from prompt
- Medium (7-8): Good balance
- High (10+): Follows prompt strictly, may become exaggerated

## Memory Optimization

### Gradient Checkpointing
Instead of saving all intermediate results in memory, we recompute them as needed. Trades computation time for VRAM.

### Attention Slicing
Splits attention computations into smaller parts. Slower but requires less VRAM.

### Gradient Accumulation
Instead of large batches (requiring lots of VRAM), we run multiple small batches and accumulate gradients.

## Results: What Has the Model Learned?

### Learned Patterns
- Color palettes from training data
- Textures and material rendering
- Composition preferences
- Form language and proportions

### Limitations
- Cannot generate exact copies of training images
- Works best with prompts similar to training data
- Quality depends on base model capabilities

## File Format

### PEFT Checkpoint
```
checkpoint/
├── adapter_config.json    # LoRA configuration
├── adapter_model.safetensors  # LoRA weights
└── README.md
```

### Safetensors
Safe binary format for model weights. Cannot contain executable code (unlike pickle).
