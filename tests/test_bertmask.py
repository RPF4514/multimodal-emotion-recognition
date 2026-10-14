"""Regression checks for contextual text masking."""
import unittest

import torch
import numpy as np

from src.data import EmotionDataset, Normalizer, deterministic_mask, reencode_masked_text
from src.explain import remove_mode


class TinyContextual(torch.nn.Module):
    def forward(self, tokens):
        ids = tokens[:, 0].float()
        attended = tokens[:, 1].float()
        context = (ids * attended).sum(1, keepdim=True)
        return context.unsqueeze(-1).expand(-1, ids.shape[1], 2)


def sample():
    ids = torch.tensor([[101, 11, 13, 17, 102, 0]])
    attended = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    return {"token_ids": ids, "text_mask": attended,
            "token_type_ids": torch.zeros_like(ids),
            "content": torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.bool),
            "valid": attended.clone(),
            "observed": attended.unsqueeze(-1).expand(-1, -1, 3).clone(),
            "teacher_text": TinyContextual()(torch.stack((ids, attended.long(),
                                                            torch.zeros_like(ids)), 1))}


class TextMaskTest(unittest.TestCase):
    def test_single_attachment_text_gets_batch_dimension(self):
        token = np.zeros((3, 50), dtype=np.int64)
        token[0, :3] = [101, 11, 102]
        token[1, :3] = 1
        record = {"text_bert": token, "text": np.ones((50, 768), dtype=np.float32),
                  "audio": np.ones((50, 74), dtype=np.float32),
                  "vision": np.ones((50, 35), dtype=np.float32), "id": "single"}
        normalizer = Normalizer(np.zeros(74), np.ones(74), np.zeros(35), np.ones(35))
        dataset = EmotionDataset(record, normalizer, labeled=False)
        self.assertEqual(tuple(dataset[0]["teacher_text"].shape), (50, 768))

    def test_recompute_changes_surviving_and_special_vectors(self):
        original = sample()
        masked = deterministic_mask(original, (0,), 0.3, position="middle")
        masked = reencode_masked_text(masked, TinyContextual())
        self.assertFalse(torch.equal(masked["teacher_text"][0, 0],
                                     original["teacher_text"][0, 0]))
        self.assertFalse(torch.equal(masked["teacher_text"][0, 1],
                                     original["teacher_text"][0, 1]))
        self.assertTrue(torch.equal(masked["teacher_text"][0, 2], torch.zeros(2)))
        self.assertTrue(torch.equal(original["token_ids"], sample()["token_ids"]))

    def test_full_modality_removes_special_tokens_too(self):
        original = sample()
        masked = remove_mode(original, 0, 0, original["valid"].shape[1], TinyContextual())
        self.assertFalse(bool(masked["observed"][0, :, 0].any()))
        self.assertFalse(bool(masked["text_mask"].any()))
        self.assertFalse(bool(masked["teacher_text"].any()))

    def test_local_explanation_recomputes_context(self):
        original = sample()
        masked = remove_mode(original, 0, 2, 3, TinyContextual())
        self.assertNotEqual(float(masked["teacher_text"][0, 0, 0]),
                            float(original["teacher_text"][0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
