"""Predict a mask for each supplied box, or boxes from an existing YOLO detector."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from core import build_model, encode, decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--decoder', required=True)
    p.add_argument('--out', default='prediction')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--box', nargs=4, type=float, action='append', metavar=('X1', 'Y1', 'X2', 'Y2'))
    source.add_argument('--yolo', help='Optional Ultralytics detection checkpoint')
    p.add_argument('--conf', type=float, default=.5)
    a = p.parse_args()
    image = np.array(Image.open(a.image).convert('RGB'))
    h, w = image.shape[:2]
    if a.yolo:
        from ultralytics import YOLO
        detector = YOLO(a.yolo)
        result = detector.predict(Image.fromarray(image), conf=a.conf, device=a.device, verbose=False)[0]
        detections = [{'box': b.xyxy[0].cpu().tolist(), 'yolo_class': result.names[int(b.cls.item())],
                       'yolo_confidence': float(b.conf.item())} for b in result.boxes]
    else:
        detections = [{'box': box} for box in a.box]
    for d in detections:
        x1, y1, x2, y2 = d['box']
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('Boxes must be nonempty xyxy coordinates within the original image')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    overlay = image.copy()
    if detections:
        sam = build_model(a.base, a.device, a.decoder)
        sam.eval()
        with torch.no_grad():
            embedding, input_size, transform = encode(sam, image)
            for i, d in enumerate(detections, 1):
                logits = decode(sam, embedding, transform, input_size, (h, w), d['box'])
                mask = logits[0, 0].cpu().numpy() > 0
                name = f'mask_{i:03d}.png'
                Image.fromarray(mask.astype(np.uint8) * 255).save(out / name)
                d.update(mask=name, area_px=int(mask.sum()), mask_source='fine-tuned SAM; estimated boundary')
                color = np.array([255, 170, 65] if d.get('yolo_class') == 'malignant' else [65, 220, 200])
                overlay[mask] = (.6 * overlay[mask] + .4 * color).astype(np.uint8)
    Image.fromarray(overlay).save(out / 'overlay.png')
    (out / 'predictions.json').write_text(json.dumps({'detections': detections,
        'notice': 'SAM estimates boundaries, not malignancy. No detection does not establish absence of disease.'}, indent=2))
    print(f'Saved {len(detections)} masks to {out}')


if __name__ == '__main__':
    main()
