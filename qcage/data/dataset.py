from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qcage.data.schema import QcageSample


class InterleavedJsonlDataset:
    """JSONL dataset for Q-CAGE samples.

    The dataset stores byte offsets instead of loading every record into memory.
    This is friendlier to large server-side training corpora.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        use_cached_features: bool = False,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.use_cached_features = use_cached_features
        self._offsets = self._build_offsets()

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        with self.jsonl_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_record(self, index: int) -> dict[str, Any]:
        with self.jsonl_path.open("rb") as handle:
            handle.seek(self._offsets[index])
            return json.loads(handle.readline().decode("utf-8"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = QcageSample.from_dict(self._read_record(index)).resolve_paths(self.image_root)
        item: dict[str, Any] = {"sample": sample}

        if self.use_cached_features:
            if sample.feature_path is None:
                raise ValueError(f"Sample {sample.sample_id} does not define feature_path")
            import torch

            features = torch.load(sample.feature_path, map_location="cpu")
            item["features"] = features

        return item

