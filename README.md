# extern_ai_collage

Training a LoRA to capture and extend an artistic visual language through machine learning.

## The Question This Project Explores

What happens when you feed 7770 images spanning 15 years of artistic practice into a neural network? Not to replicate, but to discover patterns you didn't know existed in your own work. This project documents the complete process of training a LoRA (Low-Rank Adaptation) on Stable Diffusion 1.5 using a heterogeneous archive of sculptural objects, glass works, jewelry, and mixed-media pieces.

The result is not "AI art" in the conventional sense. It's a conversation between accumulated visual decisions and mathematical pattern recognition. The model doesn't understand meaning, but it learns relationships: how certain colors cluster, how edges meet backgrounds, how light wraps around organic forms.

## What This Repository Contains

- Complete training pipeline for LoRA on SD 1.5
- Generation scripts with the exact parameters that produced good results
- 14 example images from the training dataset (artist's original works)
- Full documentation of the experimental process
- Everything needed to reproduce this workflow with your own images

## The Training Dataset

**7770 images** at 512x512 pixels, comprising:

| Category | Approximate Count | Description |
|----------|-------------------|-------------|
| Glass sculptures | ~2000 | Blown and cast glass, various scales |
| Metal/wire work | ~1500 | Copper, bronze, silver, found metal |
| Jewelry | ~1200 | Brooches, rings, pendants |
| Mixed media | ~1500 | Textile, ceramic, assemblage |
| Process documentation | ~1500 | Studio shots, work in progress |

The dataset is intentionally heterogeneous. Rather than training on a narrow, consistent style, the model learns a broader "visual fingerprint" that encompasses the full range of the practice.

**Trigger word:** `nadja_art`

All training images were captioned with: `a photo of nadja_art, [material], [object type]`

## Hardware Constraints

All training was done on a **GTX 1660 Super (6 GB VRAM)**. This required specific optimizations:

- Pre-caching all images as VAE latents to disk
- Gradient checkpointing on the UNet
- Batch size of 1 with gradient accumulation of 4
- Full precision (float32) because float16 caused NaN on this GPU
- Aggressive memory clearing between batches
- Temperature monitoring to prevent thermal throttling

## Training Configuration

These are the exact parameters used for the final successful training:

```python
CONFIG = {
    # Model
    "pretrained_model": "runwayml/stable-diffusion-v1-5",

    # LoRA architecture
    "lora_rank": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],

    # Training
    "epochs": 7,  # Sweet spot found through experimentation
    "batch_size": 1,
    "gradient_accumulation": 4,
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "scheduler": "CosineAnnealingLR",

    # Data
    "image_size": 512,
    "save_every": 1,  # Checkpoint every epoch
}
```

### Why These Values?

**LoRA rank 64:** Higher ranks capture more detail but risk overfitting. 64 was a good balance for a heterogeneous dataset.

**Learning rate 5e-5:** Lower than typical (1e-4) for better stability over many epochs. The cosine annealing helps with final convergence.

**7 epochs:** Discovered through systematic sampling. Earlier epochs produced generic outputs; later epochs started memorizing training images. Epoch 7 hit the sweet spot where the model had learned the style but still generated novel compositions.

**Gradient accumulation 4:** Effective batch size of 4 without exceeding VRAM. Larger effective batches smooth out the gradient noise.

## Generation Parameters

After extensive experimentation, these parameters produced the best results:

### For "morph" mode (subtle transformation of source images):

```python
strength = 0.2        # Only 20% transformation
guidance_scale = 7.0  # Lower = more natural
steps = 40
lora_strength = 1.55
scheduler = "DPMSolverMultistepScheduler"
vae = "stabilityai/sd-vae-ft-mse"

prompt = "nadja_art, sculptural object, dramatic lighting"
negative = "black and white, monochrome, grayscale, blurry, collage"
```

### For "transform" mode (significant style transfer):

```python
strength = 0.5        # 50% transformation
guidance_scale = 9.0  # Stronger prompt adherence
steps = 40
lora_strength = 1.55
scheduler = "DPMSolverMultistepScheduler"
vae = "stabilityai/sd-vae-ft-mse"

prompt = "nadja_art, colored glass sculpture, vibrant saturated colors, dramatic spotlight"
negative = "black and white, monochrome, grayscale, desaturated, muted colors"
```

### Critical Discoveries

1. **The VAE matters:** Using `stabilityai/sd-vae-ft-mse` instead of the default VAE significantly improved color richness.

2. **DPMSolver vs DDIM:** DPMSolverMultistepScheduler produced smoother gradients and better color transitions than DDIM.

3. **LoRA strength > 1.0:** Counterintuitively, pushing the LoRA strength to 1.55 (beyond the typical 0.5-1.0 range) produced stronger style adherence without artifacts.

4. **The trigger word is essential:** Prompts without `nadja_art` produced generic Stable Diffusion outputs. The trigger word activates the learned associations.

## The Experimental Journey

### Phase 1: Initial Failures

First attempts used typical LoRA training parameters (rank 4, lr 1e-4, 50 epochs). Results were either:
- Generic SD outputs (undertrained)
- Exact copies of training images (overtrained)

### Phase 2: Discovering the Strength Parameter

Realized that `strength` in img2img pipelines controls how much the seed affects output:
- `strength=0.2`: Nearly identical outputs regardless of seed
- `strength=0.5`: Good variation while maintaining style
- `strength=0.63`: Heavy transformation, more AI artifacts

### Phase 3: The Magic Seeds

Some seeds consistently produced better results across different source images. Documented magic seeds: `213197`, `622801`, `442076`.

This suggests these seeds traverse "favorable regions" of the latent space for this particular LoRA.

### Phase 4: Epoch Sampling

Generated 100 images from each training epoch (1-7) with identical parameters. Visual comparison revealed:

| Epoch | Character |
|-------|-----------|
| 1-2 | Generic, SD-like outputs |
| 3-4 | Style emerging, some artifacts |
| 5-6 | Strong style, good diversity |
| 7 | Peak quality, novel compositions |
| 8+ | Overfitting begins, less diversity |

## Dependencies

All packages with tested versions:

```
torch==2.1.0+cu121
torchvision==0.16.0+cu121
diffusers==0.24.0
transformers==4.36.0
accelerate==0.25.0
peft==0.7.1
safetensors==0.4.1
Pillow==10.1.0
numpy==1.26.2
tqdm==4.66.1
```

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## File Structure

```
extern_ai_collage/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
│
├── scripts/
│   ├── train_lora.py              # Training script with latent caching
│   ├── prepare_dataset.py         # Dataset preparation and captioning
│   ├── generate_simple.py         # Minimal prompt generation
│   ├── generate_100_random.py     # Structured random generation
│   └── generate_images.py         # Basic generation script
│
├── config/
│   └── training_config.yaml       # Configuration template
│
├── examples/
│   ├── dataset_sample/            # 14 images from training data
│   ├── generated/                 # Example generated outputs
│   └── generated_simple/          # Minimal prompt outputs
│
└── docs/
    ├── EXPERIMENT_LOG.md          # Detailed experiment history
    ├── PARAMETERS.md              # All tested parameter combinations
    └── PHILOSOPHY.md              # Artistic rationale
```

## Quick Start

### 1. Prepare Your Dataset

Create a folder with your images and run:

```bash
python scripts/prepare_dataset.py \
    --input /path/to/your/images \
    --output ./my_dataset \
    --token my_style \
    --size 512
```

This creates `metadata.jsonl` with captions in format: `a photo of my_style, [detected content]`

### 2. Train LoRA

```bash
python scripts/train_lora.py \
    --dataset ./my_dataset \
    --output ./training_output \
    --epochs 7 \
    --rank 64 \
    --lr 5e-5
```

Training time: ~2.5 hours per epoch on GTX 1660 Super (7770 images).

### 3. Generate Images

```bash
# Simple generation (just the trigger word)
python scripts/generate_simple.py \
    --checkpoint ./training_output/checkpoints/epoch_007 \
    --output ./generated \
    --token my_style \
    --count 20

# Structured generation (material/object combinations)
python scripts/generate_100_random.py \
    --checkpoint ./training_output/checkpoints/epoch_007 \
    --output ./generated \
    --token my_style
```

## Philosophical Notes

### On Authorship

The generated images are neither "by the AI" nor "by the artist" in any simple sense. They emerge from:
- 15 years of accumulated visual decisions (the training data)
- Billions of images that trained the base model (Stable Diffusion)
- Mathematical transformations that find patterns in both
- Random seeds that navigate the resulting space

### On "Style"

What the LoRA learns is not style in the art-historical sense. It learns statistical regularities: color distributions, edge frequencies, spatial relationships. That these regularities happen to correspond to what humans perceive as "style" is remarkable but not magical.

### On Usefulness

These tools are most valuable not for producing finished works, but for:
- Discovering unconscious patterns in your own practice
- Rapid ideation and exploration
- Understanding what makes your work visually distinctive
- Generating starting points for physical realization

## Known Limitations

1. **Memory constraints:** 6 GB VRAM limits image size and batch size
2. **Training time:** ~17 hours for full 7-epoch training
3. **Color accuracy:** Some color shifts occur, especially in saturated regions
4. **Composition:** Model sometimes produces floating fragments instead of coherent objects
5. **Detail:** Fine details (textures, small elements) often get smoothed out

## License

MIT License. The code is free to use and modify.

Note: The example images in `examples/dataset_sample/` are copyrighted artwork and are included only for demonstration purposes.

## Acknowledgments

- Stability AI for Stable Diffusion
- Hugging Face for diffusers and PEFT libraries
- Microsoft Research for the LoRA technique
- The entire open-source ML community

---

*This project was developed as part of exploring the intersection between traditional craft practice and machine learning, documenting both the technical process and the philosophical questions it raises.*
