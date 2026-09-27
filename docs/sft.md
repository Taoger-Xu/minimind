# MiniMind SFT 训练记录

## 1. SFT 的目的

预训练让模型学习语言规律和通用知识，但它本质上只是在预测下一个 token，尚未充分学会按照用户指令回答。监督微调（Supervised Fine-Tuning，SFT）使用高质量对话作为标准答案，使模型学习：

- 区分 system、user、assistant 等角色；
- 根据用户指令生成回答；
- 按聊天模板输出内容并在合适的位置结束；
- 只模仿 assistant 的回答，而不是预测用户输入。

本次 SFT 从已经完成的预训练权重 `pretrain_full_gpu0_768.pth` 出发，对同一个 63.91M 参数的 Dense 模型继续训练。模型架构没有改变，改变的是训练数据格式、label 掩码和学习率。

## 2. 本次训练结果

- 状态：已完成（2026-09-27 23:29，Asia/Shanghai）
- 数据集：`dataset/sft_t2t.jsonl`，约 14 GiB、5,109,432 条对话
- 初始权重：`out/pretrain_full_gpu0_768.pth`
- 模型：63.91M 参数，`hidden_size=768`，8 层 Transformer，非 MoE
- 设备：4 × NVIDIA A100-SXM4-80GB
- 训练轮数：2 Epoch
- 每轮步数：79,835
- 最终位置：Epoch 2，step 79,835/79,835
- 最终 loss：1.3463
- 总耗时：约 8 小时 21 分钟（包含首次数据解析和编译）

## 3. 数据格式与处理

SFT 数据的核心字段是 `conversations`，每条样本包含一轮或多轮消息。结构大致如下：

```json
{
  "conversations": [
    {"role": "user", "content": "请解释什么是机器学习"},
    {"role": "assistant", "content": "机器学习是……"}
  ]
}
```

实际数据还可以包含 `system`、`reasoning_content`、`tools` 和 `tool_calls`。对应实现位于 `dataset/lm_dataset.py` 的 `SFTDataset`。

单条样本的处理过程：

1. `datasets.load_dataset("json")` 从 JSONL 读取 `conversations`。
2. 如果样本没有 system 消息，`pre_processing_chat` 有 20% 概率添加一条随机 system prompt；工具调用样本保持不变。
3. `tokenizer.apply_chat_template` 将多角色消息转换成完整对话文本。
4. `post_processing_chat` 对空的 `<think>` 标签做随机清理，80% 概率移除空标签。
5. Tokenizer 将文本转换成 token，并截断或填充到 768 token。
6. `generate_labels` 默认把全部位置设为 `-100`，只把 assistant 回答及其结束标记对应的位置设置成真实 token id。

最终每条样本返回：

```text
input_ids: [768]
labels:    [768]
```

其中 `labels == -100` 的 system、user 和 padding 位置会被交叉熵忽略。因此模型可以读取完整上下文，但只有 assistant 回答会产生监督信号。

## 4. 从 Batch 到参数更新

本次每张 GPU 的 DataLoader batch size 为 16。DDP 使用四个独立进程和 `DistributedSampler`，把数据分给四张卡：

```text
单卡 input_ids: [16, 768]
单卡 labels:    [16, 768]
全局 batch size: 16 × 4 × 1（梯度累积）= 64
```

一次训练迭代的主要过程如下：

1. 将 `input_ids` 和 `labels` 移到当前 GPU。
2. 模型执行 `model(input_ids, labels=labels)`，得到 logits 和语言模型 loss。
3. 自回归训练将当前位置的 logits 与下一个 token 的 label 对齐。
4. 只在 `labels != -100` 的 assistant token 上计算交叉熵。
5. Dense 模型没有 MoE 路由损失，因此本次 `aux_loss=0`，总 loss 等于 `logits_loss`。
6. BF16 autocast 执行混合精度前向计算。
7. `loss.backward()` 计算梯度，DDP 自动在四张 GPU 之间同步梯度。
8. 梯度范数裁剪到 1.0，AdamW 更新参数，然后清空梯度。
9. 学习率通过 `get_lr` 从 `1e-5` 逐步衰减到约 `1e-6`。

可概括为：

```text
JSONL 对话
  → Chat Template
  → Tokenize、截断和 Padding
  → 构造只监督 assistant 的 labels
  → DataLoader + DDP 组成四卡 Batch
  → Transformer 前向传播
  → assistant token 交叉熵
  → 反向传播与四卡梯度同步
  → 梯度裁剪
  → AdamW 更新参数
```

## 5. 本次训练配置

| 配置 | 值 |
|---|---:|
| GPU 数量 | 4 |
| 每卡 batch size | 16 |
| 梯度累积 | 1 |
| 全局有效 batch size | 64 |
| 最大序列长度 | 768 |
| Epoch | 2 |
| 初始学习率 | 1e-5 |
| 优化器 | AdamW |
| 混合精度 | bfloat16 |
| 梯度裁剪 | 1.0 |
| DataLoader workers | 每进程 4 |
| 日志间隔 | 20 step |
| 保存间隔 | 5000 step |
| `torch.compile` | 开启 |
| SwanLab | offline |

## 6. 启动命令

训练脚本中的检查点目录使用相对路径，因此工作目录应为 `trainer/`：

```bash
cd /home/jk/work/minimind/trainer

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SWANLAB_MODE=offline \
PYTHONUNBUFFERED=1 \
../.venv/bin/torchrun --standalone --nproc_per_node=4 train_full_sft.py \
  --data_path ../dataset/sft_t2t.jsonl \
  --save_dir ../out \
  --save_weight full_sft_full \
  --from_weight pretrain_full_gpu0 \
  --from_resume 0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 768 \
  --epochs 2 \
  --batch_size 16 \
  --accumulation_steps 1 \
  --learning_rate 1e-5 \
  --dtype bfloat16 \
  --num_workers 4 \
  --grad_clip 1.0 \
  --log_interval 20 \
  --save_interval 5000 \
  --use_compile 1 \
  --use_wandb \
  --wandb_project MiniMind-Full-SFT \
  >> ../logs/full_sft_4gpu_20260927.log 2>&1
```

首次启动时，Hugging Face Datasets 会把约 14 GiB JSONL 转换并缓存成 Arrow 数据。本次约 511 万条数据的首次解析耗时约 2 分钟；随后 `torch.compile` 还需要一次编译。这个阶段 GPU 利用率较低是正常现象。

## 7. 日志、检查点与恢复

```text
训练日志：       logs/full_sft_4gpu_20260927.log
最终推理权重：   out/full_sft_full_768.pth
完整恢复检查点： checkpoints/full_sft_full_768_resume.pth
SwanLab 离线记录：trainer/swanlog/run-20260927_150838-bdivuzzpqk51fpkh1dnmm
```

- 推理权重约 132 MiB，只保存 FP16 模型参数，适合加载和推理。
- 完整检查点约 619 MiB，还保存 AdamW、scaler、epoch 和 step，适合断点续训。
- 每 5000 step 以及每个 Epoch 末尾保存一次。

如果训练意外中断，将原命令中的以下参数改为：

```bash
--from_resume 1
```

脚本会从 `checkpoints/full_sft_full_768_resume.pth` 恢复模型、优化器和训练位置。恢复时必须继续使用相同的模型结构和主要训练配置。

日志中的指标含义：

- `loss`：总损失；
- `logits_loss`：assistant token 的语言模型交叉熵；
- `aux_loss`：MoE 路由辅助损失，本次 Dense 模型恒为 0；
- `lr`：当前学习率；
- `epoch_time`：按照当前平均速度估算的本轮剩余时间。

四卡训练时，日志中的三种 loss 会先在所有 rank 间求平均，再由主进程打印和写入 SwanLab。

## 8. SwanLab 曲线

训练使用 SwanLab 离线记录。同步到网页后，会自动生成 `loss`、`logits_loss`、`aux_loss`、`learning_rate` 和 `epoch_time` 曲线：

```bash
cd /home/jk/work/minimind
source .venv/bin/activate
swanlab login
swanlab sync trainer/swanlog/run-20260927_150838-bdivuzzpqk51fpkh1dnmm
```

本次 `loss` 与 `logits_loss` 基本相同，`aux_loss` 为 0，这是因为 `use_moe=0`。单个 batch 的 loss 会因样本难度不同而波动，应重点观察平滑后的整体趋势，而不是要求每一步都下降。

## 9. 与预训练的主要区别

| 项目 | 预训练 | SFT |
|---|---|---|
| 输入数据 | 普通文本 `text` | 多角色对话 `conversations` |
| 初始参数 | 可从零开始 | 通常从预训练权重开始 |
| 监督范围 | 除 PAD 外的全部 token | 只有 assistant 回答 token |
| 训练目标 | 学习语言与知识 | 学习遵循指令和对话格式 |
| 学习率 | 本次为 `5e-4` | 本次为 `1e-5` |
| 序列长度 | 本次为 380 | 本次为 768 |
| 核心损失 | next-token 交叉熵 | 仍是 next-token 交叉熵 |

两者的网络前向传播、交叉熵、反向传播、AdamW 和 DDP 流程基本一致。SFT 最关键的差异不是更换模型架构，而是把对话格式化后通过 `labels=-100` 精确控制哪些 token 参与训练。

## 10. 本次运行中的温度保护

训练过程中 GPU 0 多次短暂达到 81～82°C。系统服务 `gpu-temp-guard` 的阈值为 80°C；温度超过阈值时，它会向该 GPU 上的训练进程发送 `SIGSTOP`，暂停 60 秒后自动发送 `SIGCONT` 恢复。

因此监控时曾出现：

- GPU 0 利用率暂时为 0%；
- 主进程状态为 `T`；
- 其他三个 DDP rank 保持运行或等待同步；
- 一分钟后训练从原 step 继续，没有丢失进度。

这些保护性暂停延长了总训练时间，但没有出现硬件热降频、CUDA OOM、NaN 或训练进程崩溃。

## 11. 训练后下一步

SFT 完成只说明优化过程正常结束，还需要验证模型的实际回答质量：

1. 严格加载 `out/full_sft_full_768.pth`，确认模型结构与权重完全匹配。
2. 使用多种中文、英文和多轮问题进行生成测试。
3. 检查回答是否能正确结束，是否存在严重重复、乱码或角色混乱。
4. 在独立验证集上计算 loss 或 perplexity，避免只依据训练 loss 判断效果。
5. 将 SFT 模型与预训练模型使用相同 prompts 对比，观察指令遵循能力的变化。
6. 如果后续有 chosen/rejected 偏好数据，可在该 SFT 权重基础上继续执行 DPO。
