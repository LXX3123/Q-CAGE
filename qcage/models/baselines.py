from __future__ import annotations

from dataclasses import dataclass

from qcage.models.qcage_adapter import QCAgeAdapter, QCAgeAdapterConfig


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    adapter_overrides: dict


BASELINES = {
    "full_qcage": BaselineSpec(name="full_qcage", adapter_overrides={}),
    "final_layer_only": BaselineSpec(name="final_layer_only", adapter_overrides={"selected_layers": [30]}),
    "global_pool": BaselineSpec(name="global_pool", adapter_overrides={"bottleneck": "mean_pool"}),
    "no_history_image": BaselineSpec(
        name="no_history_image",
        adapter_overrides={"source_names": ["query", "answer_text"]},
    ),
    "no_answer_text": BaselineSpec(
        name="no_answer_text",
        adapter_overrides={"source_names": ["query", "history_image"]},
    ),
}


def build_baseline_adapter(base_config: QCAgeAdapterConfig, baseline_name: str) -> QCAgeAdapter:
    if baseline_name not in BASELINES:
        raise KeyError(f"Unknown baseline: {baseline_name}")
    values = base_config.__dict__.copy()
    values.update(BASELINES[baseline_name].adapter_overrides)
    return QCAgeAdapter(QCAgeAdapterConfig(**values))

