# Dataset Placeholder

This is a placeholder for the training dataset. The actual dataset is not included for privacy reasons.

## Expected Structure

```
dataset/
├── metadata.jsonl        # Metadata and captions
└── images/               # Training images
    ├── img_000001.png
    ├── img_000002.png
    └── ...
```

## metadata.jsonl Format

Each line is a JSON object:

```json
{"file_name": "images/img_000001.png", "text": "a photo of my_style, sculpture, metal, wire"}
{"file_name": "images/img_000002.png", "text": "a photo of my_style, jewelry, beads, fabric"}
```

## Image Specifications

- **Format:** PNG
- **Size:** 256x256 pixels (square)
- **Color space:** RGB
- **Quantity:** Recommended 500-5000 images

## Creating Your Own Dataset

1. Collect images representing your style
2. Create captions for each image
3. Run `prepare_dataset.py`:

```bash
python scripts/prepare_dataset.py \
    --source_images /path/to/your/images \
    --source_captions /path/to/captions.jsonl \
    --output dataset/ \
    --trigger_word "my_style" \
    --size 256
```

## Caption Format for Source File

The source file (`source_captions.jsonl`) should have the format:

```json
{"image_id": "img_000001", "prompt": "sculpture, metal, wire"}
{"image_id": "img_000002", "prompt": "jewelry, beads, fabric"}
```

The script automatically adds the trigger word.
