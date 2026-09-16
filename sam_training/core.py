"""SAM 1 decoder fine-tuning with SAM's own resize/pad/crop transforms."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_model(base, device, decoder=None):
    # Original SAM ViT-B only. Use the official checkpoint, not YOLO best.pt.
    sam = sam_model_registry['vit_b'](checkpoint=str(base)).to(device)
    if decoder:
        saved = torch.load(decoder, map_location='cpu', weights_only=True)
        if saved['model_type'] != 'vit_b' or saved['base_sha256'] != sha256(base):
            raise ValueError('Decoder requires the same original SAM ViT-B checkpoint used for training')
        sam.mask_decoder.load_state_dict(saved['decoder'])
    sam.eval()
    for p in sam.parameters():
        p.requires_grad_(False)
    for p in sam.mask_decoder.parameters():
        p.requires_grad_(True)
    # Quality head is not calibrated by this recipe and is not used as confidence.
    for p in sam.mask_decoder.iou_prediction_head.parameters():
        p.requires_grad_(False)
    return sam


def encode(sam, image):
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    resized = transform.apply_image(image)
    tensor = torch.as_tensor(np.ascontiguousarray(resized), device=sam.device).permute(2, 0, 1)
    with torch.no_grad():
        embedding = sam.image_encoder(sam.preprocess(tensor).unsqueeze(0))
    return embedding, resized.shape[:2], transform


def decode(sam, embedding, transform, input_size, original_size, box):
    coords = transform.apply_boxes(np.asarray(box, dtype=np.float32).reshape(1, 4), original_size)
    box_tensor = torch.as_tensor(coords, device=sam.device, dtype=torch.float32)
    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(points=None, boxes=box_tensor, masks=None)
    # Do NOT use SamPredictor.predict or Sam.forward here: those disable gradients.
    low, _ = sam.mask_decoder(image_embeddings=embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense, multimask_output=False)
    return sam.postprocess_masks(low, input_size, original_size)


def loss_fn(logits, target):
    prob = logits.sigmoid()
    dice_loss = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss


def metrics(logits, target):
    pred, truth = logits > 0, target > .5
    intersection = (pred & truth).sum().item()
    total = pred.sum().item() + truth.sum().item()
    union = (pred | truth).sum().item()
    return {'dice': 2 * intersection / total if total else 1.,
            'iou': intersection / union if union else 1.}
