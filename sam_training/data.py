"""BUSI-style pairing, connected-component targets and auditable grouped splits."""
import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_pair(root, record):
    with Image.open(Path(root) / record['image']) as im:
        # Do not EXIF-transpose only one side of a paired annotation.
        image = np.array(im.convert('RGB'))
    union = np.zeros(image.shape[:2], dtype=np.uint8)
    for name in record['masks']:
        with Image.open(Path(root) / name) as im:
            mask = np.array(im.convert('RGB'))
        if mask.shape[:2] != union.shape:
            raise ValueError(f"Image/mask dimensions differ: {name}")
        if not np.all(mask == mask[..., :1]):
            raise ValueError(f"Expected grayscale binary mask: {name}")
        values = set(np.unique(mask).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f"Expected binary values 0/1 or 0/255: {name}")
        union |= (mask[..., 0] > 0).astype(np.uint8)
    return image, union


def components(mask):
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [(labels == i).astype(np.float32) for i in range(1, count)]


def box_from_mask(mask, rng=None):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError('Empty masks have no lesion box')
    h, w = mask.shape
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
    if rng is not None:
        dx, dy = max(1, .1 * (box[2] - box[0])), max(1, .1 * (box[3] - box[1]))
        box += np.array([-rng.uniform(0, dx), -rng.uniform(0, dy),
                         rng.uniform(0, dx), rng.uniform(0, dy)])
    box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
    return box


def prepare(root, groups_file=None, allow_image_split=False, seed=42):
    root = Path(root).resolve()
    group_map = {}
    if groups_file:
        with open(groups_file, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if not row['group'].strip() or row['image'] in group_map:
                    raise ValueError('Blank group or duplicate image in groups CSV')
                group_map[row['image']] = row['group'].strip()
    elif not allow_image_split:
        raise ValueError('Provide --groups CSV (image,group), or explicitly use --allow-image-split for exploratory work')
    records, hashes = [], {}
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg'} or re.search(r'_mask(?:_\d+)?$', path.stem):
            continue
        candidates = [p for p in path.parent.iterdir()
                      if p.suffix.lower() == '.png' and re.fullmatch(re.escape(path.stem) + r'_mask(?:_\d+)?', p.stem)]
        if not candidates:
            raise ValueError(f'Missing mask for {path.relative_to(root)}; do not assume missing means normal')
        rel = path.relative_to(root).as_posix()
        record = {'image': rel, 'masks': [p.relative_to(root).as_posix() for p in sorted(candidates)]}
        image, mask = read_pair(root, record)
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if groups_file and rel not in group_map:
            raise ValueError(f'Missing patient group for {rel}')
        group = group_map[rel] if groups_file else digest
        if digest in hashes and hashes[digest] != group:
            raise ValueError(f'Duplicate image assigned to different patient groups: {rel}')
        hashes[digest] = group
        record.update(group=group, image_sha256=digest, mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), instances=len(components(mask)))
        records.append(record)
    if not records:
        raise ValueError('No images found')
    groups = sorted({r['group'] for r in records})
    if len(groups) < 3:
        raise ValueError('Need at least three independent groups for train/val/test')
    random.Random(seed).shuffle(groups)
    n_hold = max(1, round(len(groups) * .15))
    splits = {g: 'test' if i < n_hold else 'val' if i < 2 * n_hold else 'train'
              for i, g in enumerate(groups)}
    for r in records:
        r['split'] = splits[r['group']]
    for split in ['train', 'val', 'test']:
        if not any(r['instances'] > 0 and r['split'] == split for r in records):
            raise ValueError(f'{split} has no positive masks. Add data or review group allocation; normal-only data cannot train this model')
    return {'schema': 1, 'seed': seed, 'split_unit': 'patient' if groups_file else 'image_content',
            'records': records}


def load_manifest(path):
    doc = json.loads(Path(path).read_text())
    seen = {}
    for r in doc['records']:
        if r['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Invalid split')
        for token in [('group', r['group']), ('image', r['image_sha256'])]:
            if token in seen and seen[token] != r['split']:
                raise ValueError('Data leakage across splits')
            seen[token] = r['split']
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='manifest.json')
    p.add_argument('--groups')
    p.add_argument('--allow-image-split', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    doc = prepare(a.data, a.groups, a.allow_image_split, a.seed)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError('Manifest already exists; choose another output instead of silently changing splits')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    for split in ['train', 'val', 'test']:
        rows = [r for r in doc['records'] if r['split'] == split]
        print(split, {'images': len(rows), 'positive_instances': sum(r['instances'] for r in rows),
                      'empty_images_skipped': sum(r['instances'] == 0 for r in rows)})
