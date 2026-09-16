import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything.modeling import Sam, ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer
from core import encode, decode, loss_fn, metrics
from data import prepare, read_pair, components, box_from_mask, load_manifest
from train import run_epoch
from audit import audit


def tiny_sam():
    # Real SAM components at smaller dimensions, no pretrained download required.
    sam = Sam(image_encoder=ImageEncoderViT(img_size=64, patch_size=16, embed_dim=64,
        depth=1, num_heads=4, out_chans=256, global_attn_indexes=(0,)),
        prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(4, 4),
            input_image_size=(64, 64), mask_in_chans=16),
        mask_decoder=MaskDecoder(transformer_dim=256, transformer=TwoWayTransformer(
            depth=1, embedding_dim=256, mlp_dim=512, num_heads=8), num_multimask_outputs=3))
    for p in sam.parameters():
        p.requires_grad_(False)
    for p in sam.mask_decoder.parameters():
        p.requires_grad_(True)
    return sam


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, empty=False):
        for i in range(12):
            image = np.random.default_rng(i).integers(0, 255, (32, 48, 3), dtype=np.uint8)
            mask = np.zeros((32, 48), dtype=np.uint8)
            if not empty:
                mask[5:20, 8:25] = 255
            Image.fromarray(image).save(self.root / f'benign ({i}).png')
            Image.fromarray(mask).save(self.root / f'benign ({i})_mask.png')

    def test_audit_counts_and_errors(self):
        self.dataset()
        result = audit(self.root)
        self.assertEqual((result['images'], result['mask_files']), (12, 12))
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['folders']['.']['valid_pairs'], 12)
        (self.root / 'benign (0)_mask.png').unlink()
        self.assertTrue(any(r['error'] == 'Missing mask' for r in audit(self.root)['errors']))

    def test_group_split_and_no_leakage(self):
        self.dataset()
        groups = self.root / 'groups.csv'
        groups.write_text('image,group\n' + ''.join(f'benign ({i}).png,p{i // 2}\n' for i in range(12)))
        doc = prepare(self.root, groups)
        sets = [{r['group'] for r in doc['records'] if r['split'] == s} for s in ['train', 'val', 'test']]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(doc))
        load_manifest(path)
        duplicate = dict(doc['records'][0], split='test' if doc['records'][0]['split'] != 'test' else 'train')
        doc['records'].append(duplicate)
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            load_manifest(path)

    def test_reject_empty_only_and_missing_masks(self):
        self.dataset(empty=True)
        with self.assertRaisesRegex(ValueError, 'no positive'):
            prepare(self.root, allow_image_split=True)
        (self.root / 'benign (0)_mask.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing mask'):
            prepare(self.root, allow_image_split=True)

    def test_multiple_components_and_binary_validation(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        Image.fromarray(image).save(self.root / 'a.png')
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[1:4, 1:5] = 1
        Image.fromarray(mask).save(self.root / 'a_mask.png')
        mask[:] = 0
        mask[10:13, 15:19] = 255
        Image.fromarray(mask).save(self.root / 'a_mask_1.png')
        row = {'image': 'a.png', 'masks': ['a_mask.png', 'a_mask_1.png']}
        _, union = read_pair(self.root, row)
        self.assertEqual(len(components(union)), 2)
        self.assertEqual(int(union.sum()), 24)
        mask[0, 0] = 128
        Image.fromarray(mask).save(self.root / 'a_mask_1.png')
        with self.assertRaisesRegex(ValueError, 'binary'):
            read_pair(self.root, row)

    def test_real_sam_gradients_and_original_image_alignment(self):
        sam = tiny_sam()
        image = np.zeros((31, 53, 3), dtype=np.uint8)
        target = np.zeros((31, 53), dtype=np.float32)
        target[5:20, 10:40] = 1
        embedding, size, transform = encode(sam, image)
        logits = decode(sam, embedding, transform, size, image.shape[:2], box_from_mask(target))
        self.assertEqual(tuple(logits.shape), (1, 1, 31, 53))
        optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=1e-3)
        parameter = sam.mask_decoder.mask_tokens.weight
        before = parameter.detach().clone()
        loss_fn(logits, torch.tensor(target)[None, None]).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in sam.mask_decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in sam.image_encoder.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, parameter))
        truth = torch.tensor(target)[None, None]
        self.assertEqual(metrics(truth * 20 - 10, truth), {'dice': 1., 'iou': 1.})

    def test_train_val_epoch_and_changed_annotation(self):
        self.dataset()
        doc = prepare(self.root, allow_image_split=True)
        sam = tiny_sam()
        row = doc['records'][0]
        opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=1e-4)
        scores = run_epoch(sam, [row], self.root, random.Random(1), opt)
        self.assertTrue(np.isfinite(scores['loss']))
        with torch.no_grad():
            val = run_epoch(sam, [row], self.root, random.Random(1))
        self.assertEqual(val['instances'], 1)
        Image.fromarray(np.zeros((32, 48), dtype=np.uint8)).save(self.root / row['masks'][0])
        with self.assertRaisesRegex(ValueError, 'changed'):
            run_epoch(sam, [row], self.root, random.Random(1))


if __name__ == '__main__':
    unittest.main()
