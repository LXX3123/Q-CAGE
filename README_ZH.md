# Q-CAGE 代码说明

本目录实现论文中的 Q-CAGE 方法：
Query-Guided Context Alignment for Interleaved Image-Text Generation。

代码面向 Linux GPU 服务器编写，尤其是单机多卡或多机多卡 A100 80G 环境。
当前 Windows 机器主要用于编辑和检查代码；真正训练、推理和评测建议在 Linux 服务器上执行。

## 可训练模块

本工程遵循论文中的 frozen-backbone 设置：
Qwen3-VL 和 FLUX/DiT 保持冻结，只训练 Q-CAGE 接口部分。

实际可训练参数包括：

- 三源/多层投影模块：current query、historical images、answer-side text；layer 18 / 24 / 30。
- learnable query bank 与 cross-attention bottleneck。
- conditioning MLP：把 Q-CAGE tokens 映射到 FLUX/DiT 的条件维度。

冻结部分包括：

- Qwen3-VL backbone。
- FLUX/DiT transformer 参数。
- VAE。
- text encoder。

注意：FLUX 参数冻结，但训练时不能把 FLUX forward 整体放进 `torch.no_grad()`，否则 loss 无法反传到 Q-CAGE condition tokens。

## 推荐服务器流程

进入代码目录并安装依赖：

```bash
cd code
bash scripts/setup_linux.sh
python scripts/env_check.py
```

先跑一个极小 mock 流程，确认环境、训练入口、checkpoint 和推理入口都能通：

```bash
bash scripts/make_mock_data.sh configs/mock_tiny.yaml
python -m qcage.training.train_adapter --config configs/mock_tiny.yaml
python -m qcage.inference.run_trajectory --config configs/mock_tiny.yaml
```

然后缓存 Qwen3-VL hidden states 和三源 token masks：

```bash
bash scripts/cache_vlm_features.sh \
  configs/train_a100_bf16.yaml \
  cached_features_qwen \
  data/train_with_qwen_features.jsonl
```

接着预计算 FLUX 训练所需张量，包括 prompt embeddings、VAE latents、noisy latents 和 velocity targets：

```bash
bash scripts/precompute_flux_features.sh \
  configs/train_flux_precomputed_a100.yaml \
  data/train_with_qwen_features.jsonl \
  cached_features_flux \
  data/train_precomputed_flux.jsonl
```

训练前检查 feature 文件是否完整：

```bash
bash scripts/validate_features.sh \
  configs/train_flux_precomputed_a100.yaml \
  data/train_precomputed_flux.jsonl \
  1
```

最后用多卡训练 Q-CAGE：

```bash
bash scripts/train_qcage.sh configs/train_flux_precomputed_a100.yaml 8
```

其中最后的 `8` 表示单机 8 张 GPU。若服务器 GPU 数不同，改成对应数量即可。

## 推理

普通推理入口：

```bash
bash scripts/infer_trajectory.sh configs/infer_qcage.yaml
```

如果使用 diffusers FLUX pipeline，并且已经缓存 prompt embeddings：

```bash
bash scripts/infer_trajectory.sh configs/infer_flux_precomputed.yaml
```

推理输出默认写入 config 中的 `inference.output_dir`，并生成 `manifest.jsonl`。

## OpenING / IntJudge 评测

如果使用官方 `OpenING-main/OpenING-benchmark/test_data.jsonl`，可以先转换成 Q-CAGE JSONL：

```bash
bash scripts/convert_opening_json.sh \
  ../OpenING-main/OpenING-benchmark/test_data.jsonl \
  data/opening_eval.jsonl
```

批量跑 benchmark。脚本会同时生成 Q-CAGE manifest 和官方 `MODEL_output` 格式：

```bash
bash scripts/run_opening_benchmark.sh configs/benchmark_opening.yaml 8
```

把 `Q-CAGE_output` 安装到官方 OpenING Arena 目录：

```bash
bash scripts/opening_install_outputs.sh \
  outputs/opening_qcage/Q-CAGE_output \
  ../OpenING-main
```

注册 Q-CAGE，并采样官方 `data_instance_modelAB.json` 风格的 pair 文件：

```bash
bash scripts/opening_prepare_arena.sh \
  ../OpenING-main \
  Q-CAGE \
  ../OpenING-main/OpenING-benchmark/test_data.jsonl \
  ../OpenING-main/Interleaved_Arena/qcage_pairs.json \
  GPT-4o+DALL-E3 Gemini1.5+Flux VILA-U Emu3 SEED-X
```

得到人工或模型 judge 的 JSONL/CSV 结果后，汇总 FDT、w/o Tie、w/Tie(0)、w/Tie(.5)：

```bash
bash scripts/summarize_intjudge.sh outputs/opening_qcage/judgements.csv qcage
```

说明：代码可以导出官方 `MODEL_output` 文件夹、生成 model-A/model-B pair 文件、读取 judge 结果并汇总指标；真实人工偏好或 IntJudge/GPT judge 判定本身仍由 OpenING 官方评测脚本产生。

## 数据格式

训练和评测数据使用 JSONL。每一行是一个样本：

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

字段说明：

- `sample_id`：样本唯一 ID。
- `history`：当前轮之前的多模态历史，按自然交互顺序排列。
- `query`：当前轮用户请求，可包含文本和可选图片。
- `target_image`：训练目标图像。
- `answer_text`：冻结 VLM 产生或预先提供的 answer-side text。
- `feature_path`：缓存后的 `.pt` 特征文件路径。

当 `feature_path` 存在时，训练脚本会直接读取缓存特征，避免每一步重复跑 Qwen 和 FLUX 预处理。

## Feature 缓存格式

一个完整的 `.pt` feature 文件应包含：

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
    "prompt_embeds": FloatTensor[num_prompt_tokens, d_dit],
    "pooled_prompt_embeds": FloatTensor[..., d_pooled],
    "txt_ids": FloatTensor[num_prompt_tokens, id_dim],
    "img_ids": FloatTensor[num_latent_tokens, id_dim],
    "target_latents": FloatTensor[..., latent_shape],
    "noisy_latents": FloatTensor[..., latent_shape],
    "velocity_target": FloatTensor[..., latent_shape],
    "timesteps": FloatTensor[1] or FloatTensor[],
}
```

Q-CAGE adapter 本身只需要：

- `hidden_states`
- `source_masks`

训练 FLUX flow-matching loss 时还需要：

- `prompt_embeds`
- `pooled_prompt_embeds`
- `txt_ids`
- `img_ids`
- `noisy_latents`
- `velocity_target`
- `timesteps`

## 关键配置

主要配置文件：

- `configs/qcage_base.yaml`：基础配置。
- `configs/train_a100_bf16.yaml`：A100 bf16 训练配置。
- `configs/train_flux_precomputed_a100.yaml`：推荐真实训练配置，使用预计算 Qwen+FLUX tensors。
- `configs/infer_qcage.yaml`：推理配置。
- `configs/benchmark_opening.yaml`：benchmark 配置。
- `configs/mock_tiny.yaml`：极小 mock 测试配置。

消融配置：

- `configs/ablations/no_layer18.yaml`
- `configs/ablations/no_layer24.yaml`
- `configs/ablations/no_layer30.yaml`
- `configs/ablations/final_layer_only.yaml`
- `configs/ablations/global_pool.yaml`
- `configs/ablations/no_history_image.yaml`
- `configs/ablations/no_answer_text.yaml`

Query budget 配置：

- `configs/budgets/k32.yaml`
- `configs/budgets/k64.yaml`
- `configs/budgets/k96.yaml`
- `configs/budgets/k128.yaml`

## 生产环境注意事项

Qwen feature cache 使用内部 marker 恢复 token span。
大规模训练前，建议先抽查几个缓存文件：

```bash
bash scripts/validate_features.sh \
  configs/train_flux_precomputed_a100.yaml \
  data/train_precomputed_flux.jsonl \
  1
```

对于包含对应来源的样本，`query`、`history_image`、`answer_text` 的 mask count 应该非零。

FLUX 预计算脚本会尽量使用 diffusers pipeline 提供的：

- `encode_prompt`
- VAE encode
- latent packing helper
- image id helper

如果你的 FLUX.2 [klein] checkpoint 使用自定义 pipeline class，需要在配置中修改：

```yaml
model:
  generator:
    pipeline_class: FluxPipeline
```

并先用少量样本跑通缓存和训练 dry-run，再正式启动多卡训练。
