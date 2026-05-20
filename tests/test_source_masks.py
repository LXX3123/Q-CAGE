from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class SourceMaskTest(unittest.TestCase):
    def test_empty_masks_are_made_safe(self) -> None:
        from qcage.models.projectors import ensure_nonempty_memory

        memory = torch.randn(2, 5, 8)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        _, safe_mask = ensure_nonempty_memory(memory, mask)
        self.assertTrue(safe_mask[:, 0].all())


if __name__ == "__main__":
    unittest.main()
