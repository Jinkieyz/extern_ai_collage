#!/usr/bin/env python3
"""
prepare_dataset.py - Prepare dataset for LoRA training

Takes images and captions and creates a training-ready dataset
with correct structure and metadata.

Usage:
    python prepare_dataset.py \
        --source_images /path/to/images \
        --source_captions /path/to/captions.jsonl \
        --output /path/to/output \
        --trigger_word "my_style" \
        --size 256
"""

import argparse
import json
import shutil
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for LoRA training')
    parser.add_argument('--source_images', type=Path, required=True,
                        help='Folder with source images')
    parser.add_argument('--source_captions', type=Path, required=True,
                        help='JSONL file with captions')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output folder for dataset')
    parser.add_argument('--trigger_word', type=str, default='my_style',
                        help='Trigger word for the style')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size (square)')
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)
    print(f"Images: {args.source_images}")
    print(f"Captions: {args.source_captions}")
    print(f"Output: {args.output}")
    print(f"Trigger: {args.trigger_word}")
    print(f"Size: {args.size}x{args.size}")
    print()

    # Load captions
    print("Loading captions...")
    captions = {}
    with open(args.source_captions, 'r') as f:
        for line in f:
            item = json.loads(line)
            image_id = item['image_id']
            prompt = item.get('prompt', '')
            if prompt:
                captions[image_id] = prompt

    print(f"Found {len(captions)} captions")

    # Create output folder
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(exist_ok=True)
    img_dir = args.output / 'images'
    img_dir.mkdir(exist_ok=True)

    # Process images
    metadata = []
    processed = 0
    skipped = 0

    for img_file in sorted(args.source_images.glob('*.png')):
        image_id = img_file.stem

        if image_id not in captions:
            skipped += 1
            continue

        try:
            # Load and resize
            with Image.open(img_file) as img:
                img = img.convert('RGB')

                # Crop to square
                w, h = img.size
                size = min(w, h)
                left = (w - size) // 2
                top = (h - size) // 2
                img = img.crop((left, top, left + size, top + size))
                img = img.resize((args.size, args.size), Image.LANCZOS)

                # Save
                dst = img_dir / f"{image_id}.png"
                img.save(dst, 'PNG')

            # Create caption with trigger
            original_caption = captions[image_id]
            full_caption = f"a photo of {args.trigger_word}, {original_caption}"

            metadata.append({
                "file_name": f"images/{image_id}.png",
                "text": full_caption
            })
            processed += 1

            if processed % 500 == 0:
                print(f"  Processed {processed}...")

        except Exception as e:
            print(f"  Error with {img_file.name}: {e}")
            skipped += 1

    # Save metadata
    meta_file = args.output / 'metadata.jsonl'
    with open(meta_file, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')

    # Show examples
    print()
    print("Example captions:")
    for item in metadata[:5]:
        print(f"  {item['text']}")

    print()
    print("=" * 60)
    print("DONE!")
    print(f"Processed images: {processed}")
    print(f"Skipped (no caption): {skipped}")
    print(f"Metadata: {meta_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
