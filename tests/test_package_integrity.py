"""Check packaged artifacts against the same guards used by inference/audit."""
from pathlib import Path
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch
from src.paths import PROJECT
from src.fetch_verified_bert import FILES, sha256
from src.evaluate_locked_bertmask_test_20260925 import (
    LOCKED_CHECKPOINT_SHA256, verify_locked_checkpoint,
)


class PackageIntegrityTest(TestCase):
    def test_packaged_bert_config_matches_public_file_hash(self):
        path = PROJECT / "outputs/bert/bert-base-uncased/config.json"
        self.assertEqual(sha256(path), FILES["config.json"])

    def test_packaged_checkpoint_passes_actual_guard(self):
        path = PROJECT / "outputs/checkpoints/gated_bigru_teacher_bertmask24_seed43.pt"
        self.assertEqual(verify_locked_checkpoint(path), LOCKED_CHECKPOINT_SHA256)

    def test_corrupted_checkpoint_is_rejected(self):
        with patch.object(Path, "open", return_value=BytesIO(b"not the locked checkpoint")):
            with self.assertRaisesRegex(ValueError, "hash changed"):
                verify_locked_checkpoint(Path("corrupted.pt"))
