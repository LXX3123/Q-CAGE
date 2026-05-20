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
class LossTest(unittest.TestCase):
    def test_flow_interpolation(self) -> None:
        from qcage.training.loss import interpolate_flow_latent

        z0 = torch.zeros(2, 3)
        z1 = torch.ones(2, 3)
        tau = torch.full((2,), 0.25)
        noisy, velocity = interpolate_flow_latent(z0, z1, tau)
        self.assertTrue(torch.allclose(noisy, torch.full((2, 3), 0.25)))
        self.assertTrue(torch.allclose(velocity, torch.ones(2, 3)))


if __name__ == "__main__":
    unittest.main()
