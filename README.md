# Q-CAGE Codebase

This folder contains the implementation for the thesis method:
Query-Guided Context Alignment for Interleaved Image-Text Generation.

The code is written for Linux GPU servers, especially multi-node or single-node
multi-GPU A100 machines. It is still importable on Windows for editing, but the
training scripts and shell wrappers assume a Linux environment.

## What Is Trainable

The frozen-backbone setting is implemented by keeping Qwen3-VL and FLUX/DiT
parameters frozen while training only the Q-CAGE interface:

- source/depth projection blocks
- learnable query bank and cross-attention bottleneck
- conditioning MLP that maps Q-CAGE tokens into the generator conditioning width

## Typical Server Workflow

```bash
cd code
bash scripts/setup_linux.sh
python scripts/env_check.py

# Fast dry run with a tiny synthetic dataset and mock generator.
bash scripts/make_mock_data.sh configs/mock_tiny.yaml
python -m qcage.training.train_adapter --config configs/mock_tiny.yaml
python -m qcage.inference.run_trajectory --config configs/mock_tiny.yaml

# Optional but recommended for expensive VLM backbones.
bash scripts/cache_vlm_features.sh \
  configs/train_a100_bf16.yaml \
  cached_features_qwen \
  data/train_with_qwen_features.jsonl

# Precompute FLUX prompt embeddings, VAE latents, noisy latents, and velocity targets.
bash scripts/precompute_flux_features.sh \
  configs/train_flux_precomputed_a100.yaml \
  data/train_with_qwen_features.jsonl \
  cached_features_flux \
  data/train_precomputed_flux.jsonl

# Multi-GPU training from precomputed Qwen+FLUX tensors.
bash scripts/validate_features.sh configs/train_flux_precomputed_a100.yaml data/train_precomputed_flux.jsonl 1
bash scripts/train_qcage.sh configs/train_flux_precomputed_a100.yaml 8

# Run turn-level or trajectory-level inference.
bash scripts/infer_trajectory.sh configs/infer_qcage.yaml

# Inference through a diffusers FLUX pipeline with cached prompt embeddings.
bash scripts/infer_trajectory.sh configs/infer_flux_precomputed.yaml

# Convert the official OpenING split to Q-CAGE JSONL if needed.
bash scripts/convert_opening_json.sh \
  ../OpenING-main/OpenING-benchmark/test_data.jsonl \
  data/opening_eval.jsonl

# Export outputs in both Q-CAGE manifest and official OpenING MODEL_output format.
bash scripts/run_opening_benchmark.sh configs/benchmark_opening.yaml 8

# Install Q-CAGE_output into the official OpenING arena directory.
bash scripts/opening_install_outputs.sh \
  outputs/opening_qcage/Q-CAGE_output \
  ../OpenING-main

# Register Q-CAGE and sample official model-A/model-B pairs.
bash scripts/opening_prepare_arena.sh \
  ../OpenING-main \
  Q-CAGE \
  ../OpenING-main/OpenING-benchmark/test_data.jsonl \
  ../OpenING-main/Interleaved_Arena/qcage_pairs.json \
  GPT-4o+DALL-E3 Gemini1.5+Flux VILA-U Emu3 SEED-X

# Summarize IntJudge JSONL/CSV results after judging.
bash scripts/summarize_intjudge.sh outputs/opening_qcage/judgements.csv qcage
```

For official OpenING JSONL files that do not already match the Q-CAGE schema:

```bash
bash scripts/convert_opening_json.sh \
  ../OpenING-main/OpenING-benchmark/test_data.jsonl \
  data/opening_eval.jsonl
```

`run_opening_benchmark.py` writes official files like
`outputs/opening_qcage/Q-CAGE_output/1502046.json` and
`outputs/opening_qcage/Q-CAGE_output/1502046-o-0.png`. This mirrors the
`MODELNAME_output` folders expected by OpenING's `IntJudge_judge_AB.py`.

## Data Format

Training and evaluation data can be provided as JSONL. Each line is one sample:

```json
{
  "sample_id": "case_0001_turn_03",
  "history": [
    {"role": "user", "text": "Create a traveler in a rainy old town.", "image": "images/t1.png"},
    {"role": "assistant", "text": "The traveler is near the station."}
  ],
  "query": {
    "role": "user",
    "text": "Move the same traveler to the open corner pub.",
    "image": "images/current_ref.png"
  },
  "target_image": "targets/case_0001_turn_03.png",
  "answer_text": "The open corner pub should be the next destination.",
  "feature_path": "cached_features/case_0001_turn_03.pt"
}
```

When `feature_path` is present, the training loop can skip the expensive VLM
forward and load cached hidden states/masks directly.

## Cached Feature Tensor Contract

A cached `.pt` feature file should contain:

```python
{
    "hidden_states": {
        18: FloatTensor[num_tokens, d_vlm],
        24: FloatTensor[num_tokens, d_vlm],
        30: FloatTensor[num_tokens, d_vlm],
    },
    "source_masks": {
        "query": BoolTensor[num_tokens],
        "history_image": BoolTensor[num_tokens],
        "answer_text": BoolTensor[num_tokens],
    },
    "prompt_embeds": FloatTensor[num_prompt_tokens, d_dit],      # FLUX precompute
    "pooled_prompt_embeds": FloatTensor[..., d_pooled],          # FLUX precompute
    "txt_ids": FloatTensor[num_prompt_tokens, id_dim],           # FLUX precompute
    "img_ids": FloatTensor[num_latent_tokens, id_dim],           # FLUX precompute
    "target_latents": FloatTensor[..., latent_shape],            # bridge-dependent
    "noisy_latents": FloatTensor[..., latent_shape],             # FLUX precompute
    "velocity_target": FloatTensor[..., latent_shape],           # FLUX precompute
    "timesteps": FloatTensor[1] or FloatTensor[],                # FLUX precompute
}
```

The exact latent/prompt fields depend on the FLUX bridge used in training. The
core Q-CAGE adapter only requires `hidden_states` and `source_masks`.

For `model.generator.backend=diffusers_flux_precomputed`, each feature file
should also contain `noisy_latents`, `velocity_target`, `prompt_embeds`, and
the FLUX-specific IDs/pooled embeddings required by the selected diffusers
pipeline, such as `pooled_prompt_embeds`, `txt_ids`, and `img_ids`.

## Production Notes

The Qwen feature cache uses internal source markers and then recovers token
spans from `input_ids`. Check several cached files before launching a large run:
the `query`, `history_image`, and `answer_text` masks should all have nonzero
counts for samples that contain those sources.

The FLUX precompute script uses the selected diffusers pipeline's `encode_prompt`,
VAE, latent packing helpers, and image-id helpers when available. If your
FLUX.2 [klein] checkpoint uses a custom pipeline class, set
`model.generator.pipeline_class` in the config and verify one dry-run batch
before launching multi-GPU training.
