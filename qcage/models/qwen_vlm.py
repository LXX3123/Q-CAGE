from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qcage.data.schema import QcageSample
from qcage.data.serialize import serialize_for_vlm


QUERY_START = "<QCAGE_QUERY_START>"
QUERY_END = "<QCAGE_QUERY_END>"
HISTORY_IMAGE_START = "<QCAGE_HISTORY_IMAGE_START>"
HISTORY_IMAGE_END = "<QCAGE_HISTORY_IMAGE_END>"
ANSWER_START = "<QCAGE_ANSWER_START>"
ANSWER_END = "<QCAGE_ANSWER_END>"


@dataclass
class VLMForwardOutput:
    hidden_states: dict[int, Any]
    source_masks: dict[str, Any]
    answer_text: str | None = None


class QwenVLMWrapper:
    """Frozen Qwen-VL wrapper.

    The production training path should usually cache VLM outputs first. Source
    span recovery for real multimodal tokenization is model/version specific, so
    the cache script is the right place to verify exact masks for a given Qwen
    release.
    """

    def __init__(
        self,
        model_id: str,
        selected_layers: Sequence[int],
        device: str = "cuda",
        dtype: str = "bf16",
        trust_remote_code: bool = True,
        device_map: str | dict | None = None,
        attn_implementation: str | None = None,
        generate_answer_if_missing: bool = True,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_id = model_id
        self.selected_layers = [int(layer) for layer in selected_layers]
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.generate_answer_if_missing = generate_answer_if_missing
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def load(self) -> "QwenVLMWrapper":
        import torch
        from transformers import AutoProcessor

        dtype = torch.bfloat16 if self.dtype == "bf16" else torch.float16 if self.dtype == "fp16" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        errors: list[str] = []
        for class_name in [
            "Qwen3VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]:
            try:
                import transformers

                model_cls = getattr(transformers, class_name)
                self.model = model_cls.from_pretrained(self.model_id, **model_kwargs)
                break
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        if self.model is None:
            raise RuntimeError("Could not load Qwen VLM:\n" + "\n".join(errors))

        if self.device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        return self

    def serialize_batch(self, samples: Sequence[QcageSample]) -> list[str]:
        return [serialize_for_vlm(sample).text for sample in samples]

    def _marked_messages(self, sample: QcageSample, answer_text: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        for message in sample.history:
            content: list[dict[str, str]] = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            if message.image:
                content.append({"type": "text", "text": f" {HISTORY_IMAGE_START} "})
                content.append({"type": "image", "image": message.image})
                content.append({"type": "text", "text": f" {HISTORY_IMAGE_END} "})
            if content:
                messages.append({"role": message.role, "content": content})

        query_content: list[dict[str, str]] = [{"type": "text", "text": f" {QUERY_START} "}]
        if sample.query.text:
            query_content.append({"type": "text", "text": sample.query.text})
        if sample.query.image:
            query_content.append({"type": "image", "image": sample.query.image})
        query_content.append({"type": "text", "text": f" {QUERY_END} "})
        messages.append({"role": sample.query.role, "content": query_content})

        if answer_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f" {ANSWER_START} "},
                        {"type": "text", "text": answer_text},
                        {"type": "text", "text": f" {ANSWER_END} "},
                    ],
                }
            )
        return messages

    def _inputs_from_messages(self, messages: list[dict[str, Any]], add_generation_prompt: bool):
        if self.processor is None:
            raise RuntimeError("Call load() before extracting features.")
        import torch

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as exc:
                raise RuntimeError(
                    "qwen-vl-utils is required when processor.apply_chat_template cannot "
                    "process multimodal messages directly."
                ) from exc
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(self.device)
        return inputs

    def _marker_ids(self, marker: str) -> list[int]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        token_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    @staticmethod
    def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
        if not pattern:
            return -1
        last = len(sequence) - len(pattern)
        for index in range(start, last + 1):
            if sequence[index : index + len(pattern)] == pattern:
                return index
        return -1

    def _span_mask(self, input_ids, start_marker: str, end_marker: str, include_all: bool):
        import torch

        ids = [int(item) for item in input_ids.tolist()]
        start_ids = self._marker_ids(start_marker)
        end_ids = self._marker_ids(end_marker)
        mask = torch.zeros(len(ids), dtype=torch.bool)
        offset = 0
        while offset < len(ids):
            start = self._find_subsequence(ids, start_ids, offset)
            if start < 0:
                break
            content_start = start + len(start_ids)
            end = self._find_subsequence(ids, end_ids, content_start)
            if end < 0:
                break
            mask[content_start:end] = True
            offset = end + len(end_ids)
            if not include_all:
                break
        return mask

    def _source_masks_from_input_ids(self, input_ids):
        return {
            "query": self._span_mask(input_ids, QUERY_START, QUERY_END, include_all=False),
            "history_image": self._span_mask(
                input_ids,
                HISTORY_IMAGE_START,
                HISTORY_IMAGE_END,
                include_all=True,
            ),
            "answer_text": self._span_mask(input_ids, ANSWER_START, ANSWER_END, include_all=False),
        }

    @staticmethod
    def _hidden_state_tuple(output):
        if getattr(output, "hidden_states", None) is not None:
            return output.hidden_states
        for attr in ["language_model_outputs", "model_outputs", "llm_outputs"]:
            nested = getattr(output, attr, None)
            if nested is not None and getattr(nested, "hidden_states", None) is not None:
                return nested.hidden_states
        raise RuntimeError("The Qwen forward output did not contain hidden_states.")

    def _generate_answer(self, sample: QcageSample) -> str:
        if not self.generate_answer_if_missing:
            return ""
        if self.model is None:
            raise RuntimeError("Call load() before generating answers.")
        import torch

        messages = self._marked_messages(sample, answer_text=None)
        inputs = self._inputs_from_messages(messages, add_generation_prompt=True)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        if hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        tokenizer = getattr(self.processor, "tokenizer")
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def extract(self, samples: Sequence[QcageSample]) -> VLMForwardOutput:
        if self.model is None:
            raise RuntimeError("Call load() before extracting features.")
        if len(samples) != 1:
            raise ValueError("QwenVLMWrapper.extract currently expects one sample at a time.")
        import torch

        sample = samples[0]
        answer_text = sample.answer_text or self._generate_answer(sample)
        messages = self._marked_messages(sample, answer_text=answer_text)
        inputs = self._inputs_from_messages(messages, add_generation_prompt=False)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        hidden_tuple = self._hidden_state_tuple(outputs)
        hidden_states: dict[int, Any] = {}
        for layer in self.selected_layers:
            if layer >= len(hidden_tuple):
                raise IndexError(
                    f"Requested hidden layer {layer}, but model returned only "
                    f"{len(hidden_tuple)} hidden-state tensors."
                )
            hidden_states[layer] = hidden_tuple[layer][0].detach().cpu()

        input_ids = inputs["input_ids"][0].detach().cpu()
        source_masks = self._source_masks_from_input_ids(input_ids)
        return VLMForwardOutput(
            hidden_states=hidden_states,
            source_masks=source_masks,
            answer_text=answer_text,
        )


def build_vlm_from_config(config: dict, device: str = "cuda") -> QwenVLMWrapper:
    vlm_cfg = config["model"]["vlm"]
    return QwenVLMWrapper(
        model_id=vlm_cfg["model_id"],
        selected_layers=vlm_cfg.get("selected_layers", [18, 24, 30]),
        device=device,
        dtype=config.get("training", {}).get("precision", "bf16"),
        trust_remote_code=bool(vlm_cfg.get("trust_remote_code", True)),
        device_map=vlm_cfg.get("device_map"),
        attn_implementation=vlm_cfg.get("attn_implementation"),
        generate_answer_if_missing=bool(vlm_cfg.get("generate_answer_if_missing", True)),
        max_new_tokens=int(vlm_cfg.get("max_new_tokens", 128)),
    )
