# External Dependencies and Sources

This document lists all external programs, libraries, and models used in the project.

## Base Model

### Stable Diffusion 1.5
- **Source:** [Runway ML / Stability AI](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **License:** CreativeML Open RAIL-M
- **Description:** A latent diffusion model trained on the LAION-5B dataset (5 billion text-image pairs). The model can generate images from text descriptions.
- **Components used:**
  - **UNet:** Main network performing the diffusion process (denoising)
  - **VAE (Variational Autoencoder):** Encodes images to/from latent space
  - **CLIP Text Encoder:** Converts text to vectors the model understands
  - **Scheduler:** Controls the diffusion process timesteps

## Python Libraries

### PyTorch
- **Version:** 2.x
- **Source:** [pytorch.org](https://pytorch.org/)
- **License:** BSD-3-Clause
- **Description:** Machine learning framework. Handles tensors, GPU acceleration, and automatic differentiation.
- **Usage:** Basic computations, model training, inference

### Diffusers
- **Version:** 0.25+
- **Source:** [Hugging Face](https://github.com/huggingface/diffusers)
- **License:** Apache 2.0
- **Description:** Library for diffusion models. Simplifies loading and using Stable Diffusion.
- **Usage:** Pipeline for image generation, scheduler implementations

### Transformers
- **Version:** 4.x
- **Source:** [Hugging Face](https://github.com/huggingface/transformers)
- **License:** Apache 2.0
- **Description:** Library for transformer models including CLIP.
- **Usage:** Text encoding via CLIP

### PEFT (Parameter-Efficient Fine-Tuning)
- **Version:** 0.7+
- **Source:** [Hugging Face](https://github.com/huggingface/peft)
- **License:** Apache 2.0
- **Description:** Implements LoRA and other efficient fine-tuning methods.
- **Usage:** LoRA configuration, training and loading adapters

### Accelerate
- **Version:** 0.25+
- **Source:** [Hugging Face](https://github.com/huggingface/accelerate)
- **License:** Apache 2.0
- **Description:** Simplifies distributed training and mixed precision.
- **Usage:** Gradient accumulation, training loop

### Pillow (PIL)
- **Version:** 10.x
- **Source:** [python-pillow.org](https://python-pillow.org/)
- **License:** HPND
- **Description:** Image processing library.
- **Usage:** Load, save, and manipulate images

### Safetensors
- **Version:** 0.4+
- **Source:** [Hugging Face](https://github.com/huggingface/safetensors)
- **License:** Apache 2.0
- **Description:** Safe file format for model weights.
- **Usage:** Save and load LoRA weights

## Algorithms and Methods

### LoRA (Low-Rank Adaptation)
- **Source:** [Microsoft Research](https://arxiv.org/abs/2106.09685)
- **Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **Description:** Method for fine-tuning large models by training only low-rank matrices added to existing weights. Dramatically reduces memory usage and training time.

### DDPM (Denoising Diffusion Probabilistic Models)
- **Source:** [UC Berkeley](https://arxiv.org/abs/2006.11239)
- **Paper:** "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **Description:** Fundamental diffusion method that Stable Diffusion builds upon.

### DPM-Solver++
- **Source:** [Tsinghua University](https://arxiv.org/abs/2211.01095)
- **Paper:** "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
- **Description:** Faster sampling algorithm that reduces steps from 50+ to 20-30.

### CLIP (Contrastive Language-Image Pre-training)
- **Source:** [OpenAI](https://openai.com/research/clip)
- **Paper:** "Learning Transferable Visual Models From Natural Language Supervision"
- **Description:** Model connecting text and images in a shared vector space.

### Latent Diffusion
- **Source:** [CompVis / LMU Munich](https://arxiv.org/abs/2112.10752)
- **Paper:** "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
- **Description:** Performs diffusion in compressed latent space instead of pixel space, dramatically reducing computational requirements.

## Hardware Requirements

### GPU
- **Minimum:** NVIDIA GPU with 6 GB VRAM
- **Recommended:** 8+ GB VRAM
- **Tested on:** NVIDIA GTX 1660 Super (6 GB)

### CUDA
- **Version:** 11.8 or 12.x
- **Source:** [NVIDIA](https://developer.nvidia.com/cuda-toolkit)
- **Description:** GPU acceleration for PyTorch

## License Summary

| Component | License | Commercial Use |
|-----------|---------|----------------|
| Stable Diffusion 1.5 | CreativeML Open RAIL-M | Yes, with restrictions |
| PyTorch | BSD-3-Clause | Yes |
| Diffusers | Apache 2.0 | Yes |
| Transformers | Apache 2.0 | Yes |
| PEFT | Apache 2.0 | Yes |
| Pillow | HPND | Yes |

## Citations

If using this project in academic work:

```bibtex
@misc{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and others},
  year={2021},
  eprint={2106.09685},
  archivePrefix={arXiv}
}

@misc{rombach2022highresolution,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and others},
  year={2022},
  eprint={2112.10752},
  archivePrefix={arXiv}
}
```
