#!/usr/bin/env python3
"""Deterministically crop a square avatar from a local character portrait."""
import argparse
from pathlib import Path
from PIL import Image


def crop_avatar(source, target, box):
    with Image.open(source) as image:
        left, top, right, bottom = box
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height and
                right - left == bottom - top):
            raise ValueError('Avatar crop must be a square inside the portrait')
        result = image.convert('RGBA').crop(box).resize((256, 256), Image.Resampling.LANCZOS)
        target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target, format='WEBP', lossless=True, method=6, exact=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--box', nargs=4, type=int, required=True)
    args = parser.parse_args()
    crop_avatar(args.source, args.target, tuple(args.box))
