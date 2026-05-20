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
class AdapterShapeTest(unittest.TestCase):
    def test_qcage_adapter_output_shape(self) -> None:
        from qcage.models.qcage_adapter import QCAgeAdapter, QCAgeAdapterConfig

        config = QCAgeAdapterConfig(
            vlm_hidden_dim=32,
            adapter_dim=16,
            dit_hidden_dim=64,
            selected_layers=(18, 24, 30),
            source_names=("query", "history_image", "answer_text"),
            num_queries=8,
            num_heads=4,
            cross_attention_layers=1,
        )
        model = QCAgeAdapter(config)
        batch_size, num_tokens = 2, 12
        hidden_states = {
            18: torch.randn(batch_size, num_tokens, 32),
            24: torch.randn(batch_size, num_tokens, 32),
            30: torch.randn(batch_size, num_tokens, 32),
        }
        source_masks = {
            "query": torch.zeros(batch_size, num_tokens, dtype=torch.bool),
            "history_image": torch.zeros(batch_size, num_tokens, dtype=torch.bool),
            "answer_text": torch.zeros(batch_size, num_tokens, dtype=torch.bool),
        }
        source_masks["query"][:, :4] = True
        source_masks["history_image"][:, 4:8] = True
        source_masks["answer_text"][:, 8:] = True

        output = model(hidden_states, source_masks)
        self.assertEqual(tuple(output.condition_tokens.shape), (batch_size, 8, 64))
        self.assertEqual(tuple(output.bottleneck_tokens.shape), (batch_size, 8, 16))
        self.assertTrue(output.memory_mask.any(dim=1).all())


if __name__ == "__main__":
    unittest.main()
