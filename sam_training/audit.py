"""Inspect paired masks locally before any split/training; no files are modified."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from data import read_pair, components


def audit(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('Dataset directory does not exist')
    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    masks = [p for p in files if re.search(r'_mask(?:_\d+)?$', p.stem)]
    mask_set = set(masks)
    images = [p for p in files if p not in mask_set]
    used = set()
    report = {'images': len(images), 'mask_files': len(masks), 'folders': {}, 'errors': []}
    for image in images:
        pairs = [m for m in masks if m.parent == image.parent and re.fullmatch(re.escape(image.stem) + r'_mask(?:_\d+)?', m.stem)]
        key = image.parent.relative_to(root).as_posix()
        stats = report['folders'].setdefault(key, {'images': 0, 'mask_files': 0, 'empty_masks': 0, 'instances': 0, 'valid_pairs': 0})
        stats['images'] += 1
        stats['mask_files'] += len(pairs)
        rel = image.relative_to(root).as_posix()
        if not pairs:
            report['errors'].append({'image': rel, 'error': 'Missing mask'})
            continue
        used.update(pairs)
        try:
            _, mask = read_pair(root, {'image': rel, 'masks': [p.relative_to(root).as_posix() for p in pairs]})
            stats['valid_pairs'] += 1
            stats['instances'] += len(components(mask))
            stats['empty_masks'] += int(not np.any(mask))
        except (ValueError, OSError) as exc:
            report['errors'].append({'image': rel, 'error': str(exc)})
    for p in masks:
        if p not in used:
            report['errors'].append({'mask': p.relative_to(root).as_posix(), 'error': 'Orphan mask'})
    if not images:
        report['errors'].append({'error': 'No images found'})
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    args = p.parse_args()
    report = audit(args.data)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report['errors'] else 0)
