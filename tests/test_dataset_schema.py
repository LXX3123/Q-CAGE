from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcage.data.schema import QcageSample
from qcage.data.serialize import prompt_for_generator, serialize_for_vlm


class DatasetSchemaTest(unittest.TestCase):
    def test_sample_round_trip_and_serialization(self) -> None:
        sample = QcageSample.from_dict(
            {
                "sample_id": "s1",
                "history": [{"role": "user", "text": "old text", "image": "old.png"}],
                "query": {"role": "user", "text": "new text"},
                "answer_text": "answer",
            }
        )
        self.assertEqual(sample.to_dict()["sample_id"], "s1")
        serialized = serialize_for_vlm(sample)
        self.assertIn("old text", serialized.text)
        self.assertIn("answer", serialized.text)
        self.assertEqual(prompt_for_generator(sample), "new text")


if __name__ == "__main__":
    unittest.main()
