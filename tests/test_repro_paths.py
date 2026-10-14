"""Prevent independent inference from silently loading the training checkout."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src import paths


class ReproPathTest(TestCase):
    def test_local_checkpoint_wins_over_external_absolute_path(self):
        external = paths.PROJECT.parent / "training" / "selected.pt"
        local_dir = paths.PROJECT / "outputs" / "checkpoints"
        local = local_dir / "selected.pt"
        with patch.object(paths, "CHECKPOINTS", local_dir), patch.object(
            Path, "is_file", lambda target: target in (external, local)
        ):
            self.assertEqual(
                paths.selected_checkpoint_path({"selected_checkpoint": str(external)}),
                local,
            )

    def test_missing_local_checkpoint_rejects_training_path(self):
        external = paths.PROJECT.parent / "training" / "selected.pt"
        local_dir = paths.PROJECT / "outputs" / "checkpoints"
        with patch.object(paths, "CHECKPOINTS", local_dir), patch.object(
            Path, "is_file", lambda target: target == external
        ):
            with self.assertRaises(FileNotFoundError):
                paths.selected_checkpoint_path({"selected_checkpoint": str(external)})
