"""Train decoder only, select by validation Dice; test is a separate command."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from core import build_model, encode, decode, loss_fn, metrics, sha256
from data import read_pair, components, box_from_mask, load_manifest


def run_epoch(sam, rows, root, rng, optimizer=None):
    sam.eval()
    sam.mask_decoder.train(optimizer is not None)
    order = list(rows)
    if optimizer:
        rng.shuffle(order)
    totals = {'loss': 0., 'dice': 0., 'iou': 0.}
    n = 0
    for row in order:
        image, mask = read_pair(root, row)
        # Verify file contents have not changed since split preparation.
        import hashlib
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest != row['image_sha256'] or hashlib.sha256(mask.tobytes()).hexdigest() != row['mask_sha256']:
            raise ValueError(f"Image or mask changed after manifest creation: {row['image']}")
        instances = components(mask)
        if not instances:
            continue  # no fabricated positive prompt for normal images
        embedding, input_size, transform = encode(sam, image)
        for target_np in instances:
            box = box_from_mask(target_np, rng if optimizer else None)
            target = torch.as_tensor(target_np, device=sam.device)[None, None]
            with torch.set_grad_enabled(optimizer is not None):
                logits = decode(sam, embedding, transform, input_size, image.shape[:2], box)
                loss = loss_fn(logits, target)
                if optimizer:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sam.mask_decoder.parameters(), 1.)
                    optimizer.step()
            score = metrics(logits.detach(), target)
            totals['loss'] += loss.item()
            for k in score:
                totals[k] += score[k]
            n += 1
    if not n:
        raise ValueError('No positive instances in this split')
    return {**{k: v / n for k, v in totals.items()}, 'instances': n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--base', required=True, help='Official sam_vit_b_01ec64.pth')
    p.add_argument('--out', default='runs/sam_breast')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--evaluate', help='Path to decoder checkpoint; evaluates held-out TEST split only')
    a = p.parse_args()
    if a.epochs < 1 or a.lr <= 0 or a.patience < 1:
        p.error('epochs, lr and patience must be positive')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(4)
    rng = random.Random(a.seed)
    doc = load_manifest(a.manifest)
    rows = doc['records']
    if doc['split_unit'] != 'patient':
        print('WARNING: image-only split; patient independence is not established.')
    sam = build_model(a.base, a.device, a.evaluate)
    if a.evaluate:
        with torch.no_grad():
            scores = run_epoch(sam, [r for r in rows if r['split'] == 'test'], a.data, rng)
        print(json.dumps({'test_with_ground_truth_box_prompts': scores}, indent=2))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    optimizer = torch.optim.AdamW([p for p in sam.mask_decoder.parameters() if p.requires_grad], lr=a.lr)
    base_hash = sha256(a.base)
    best, stale = -1., 0
    for epoch in range(1, a.epochs + 1):
        train = run_epoch(sam, [r for r in rows if r['split'] == 'train'], a.data, rng, optimizer)
        with torch.no_grad():
            val = run_epoch(sam, [r for r in rows if r['split'] == 'val'], a.data, rng)
        record = {'epoch': epoch, 'train': train, 'val_with_ground_truth_box_prompts': val}
        print(json.dumps(record), flush=True)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if val['dice'] > best:
            best, stale = val['dice'], 0
            torch.save({'model_type': 'vit_b', 'base_sha256': base_hash, 'epoch': epoch,
                        'val_dice': best, 'decoder': {k: v.detach().cpu() for k, v in sam.mask_decoder.state_dict().items()}},
                       out / 'best_decoder.pt')
        else:
            stale += 1
        if stale >= a.patience:
            break
    print(f'Saved {out / "best_decoder.pt"}. Keep the original SAM base checkpoint for inference.')


if __name__ == '__main__':
    main()
