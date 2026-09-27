# MiniMind DPO 原理与训练记录

## 1. DPO 在训练链路中的位置

MiniMind 的主要训练链路为：

```text
预训练 → SFT → DPO（可选）→ 评测与推理
```

- 预训练让模型学习语言规律和通用知识；
- SFT 使用标准回答训练模型遵循指令；
- DPO 使用 `chosen/rejected` 偏好对，让模型进一步倾向人类认为更好的回答。

DPO（Direct Preference Optimization，直接偏好优化）属于基于静态偏好数据的 RLHF 方法。它由带 KL 约束的强化学习目标推导而来，但训练时不需要在线采样，也不需要单独训练 Reward Model 和 Value Model，代码结构更接近普通监督学习。

## 2. 从带 KL 约束的 RLHF 到 DPO

给定 prompt $x$ 和回答 $y$，策略模型为 $\pi_\theta(y|x)$，冻结的参考模型为 $\pi_{ref}(y|x)$。RLHF 希望提高回答奖励，同时限制策略模型不要偏离参考模型太远：

$$
\max_\pi\;\mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)]
-\beta D_{KL}(\pi(\cdot|x)\|\pi_{ref}(\cdot|x))
$$

该目标的最优策略满足：

$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)
\exp\left(\frac{r(x,y)}{\beta}\right)
$$

反过来可以把奖励表示为：

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}
+\beta\log Z(x)
$$

偏好数据包含同一个 prompt 对应的较好回答 $y_w$（chosen）和较差回答 $y_l$（rejected）。使用 Bradley-Terry 偏好模型：

$$
P(y_w\succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))
$$

把奖励表达式代入后，由于 chosen 和 rejected 共享同一个 prompt，归一化项 $Z(x)$ 会相互抵消，最终得到 DPO loss：

$$
\mathcal L_{DPO}=-\mathbb E\left[
\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
\right]
$$

其中：

- $\pi_\theta$ 是正在更新的策略模型；
- $\pi_{ref}$ 是冻结的 SFT 参考模型；
- $\beta$ 控制偏好优化强度以及相对参考模型的约束；
- $\sigma$ 是 sigmoid 函数。

## 3. 为什么 DPO 仍像交叉熵训练

定义一个回答对的分类 logit：

$$
z=\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
$$

模型认为 chosen 胜过 rejected 的概率为：

$$
p=\sigma(z)
$$

偏好数据的标签恒为 chosen 获胜，即 $t=1$。二分类交叉熵因此为：

$$
\mathcal L_{BCE}=-\log p=-\log\sigma(z)
$$

这正是 DPO loss。因此更准确地说：

- SFT 是 token 级多分类交叉熵；
- DPO 是回答对级二分类交叉熵。

DPO 虽然从强化学习目标推导而来，实际训练仍然是模型前向传播、计算可微 loss、反向传播和 AdamW 更新，所以代码与 SFT 很相似。

## 4. 偏好数据格式

本次使用 README 中的主线 RLHF 数据 `dataset/dpo.jsonl`，共 17,166 条偏好对，数据来源于 DPO-En-Zh-20k 的整理版本。

每条数据大致如下：

```json
{
  "chosen": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "good answer"}
  ],
  "rejected": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "bad answer"}
  ]
}
```

`chosen` 和 `rejected` 应拥有相同的对话上下文，只在候选回答质量上存在差异。

## 5. MiniMind 的数据处理

数据处理代码位于 `dataset/lm_dataset.py` 的 `DPODataset`：

1. 分别对 chosen 和 rejected 调用 `tokenizer.apply_chat_template`；
2. 截断或填充到 `max_seq_len=1024`；
3. 分别构造自回归输入 `x=input_ids[:-1]` 和目标 `y=input_ids[1:]`；
4. 生成 `mask`，只统计 assistant 回答 token 的 log probability；
5. 每条数据返回 chosen 与 rejected 两套 `x/y/mask`。

单卡 batch size 为 4 时，DataLoader 输出的主要形状为：

```text
x_chosen:   [4, 1023]
x_rejected: [4, 1023]
y_chosen:   [4, 1023]
y_rejected: [4, 1023]
mask:       [4, 1023]
```

训练脚本把 chosen 和 rejected 沿 batch 维拼接，策略模型和参考模型每次分别处理 `[8, 1023]`，减少单独前向传播的调用次数。

## 6. MiniMind 的 DPO loss 计算

核心代码位于 `trainer/train_dpo.py`。

模型首先输出每个位置的 logits，再取得真实目标 token 的 log probability：

```python
log_probs = F.log_softmax(logits, dim=2)
token_log_probs = torch.gather(
    log_probs,
    dim=2,
    index=labels.unsqueeze(2)
).squeeze(-1)
```

随后通过 mask 只累加 assistant 回答部分：

```python
sequence_log_probs = (token_log_probs * mask).sum(dim=1)
```

分别计算策略模型与参考模型的 chosen/rejected margin：

```python
pi_logratios = chosen_policy_log_probs - rejected_policy_log_probs
ref_logratios = chosen_ref_log_probs - rejected_ref_log_probs
logits = pi_logratios - ref_logratios
loss = -F.logsigmoid(beta * logits).mean()
```

参考模型使用 `torch.no_grad()` 且 `requires_grad_(False)`，只提供固定基准；梯度只更新策略模型。

## 7. 一次参数更新的完整流程

```text
读取 chosen/rejected
  → 应用 Chat Template
  → Tokenize、截断和 Padding
  → 构造 assistant loss mask
  → 拼接 chosen 与 rejected
  → 冻结参考模型前向传播
  → 可训练策略模型前向传播
  → 计算双方的序列 log probability
  → 构造 chosen/rejected 相对 margin
  → 计算 -logsigmoid DPO loss
  → 反向传播
  → 四卡 DDP 同步梯度
  → 梯度裁剪
  → AdamW 更新策略模型
```

本次为 Dense 模型，`aux_loss=0`，因此总 loss 等于 DPO loss。

## 8. 本次训练配置

| 配置 | 值 |
|---|---:|
| 初始权重 | `full_sft_full_768.pth` |
| 偏好数据量 | 17,166 对 |
| GPU 数量 | 4 |
| 每卡 batch size | 4 |
| 全局 batch size | 16 |
| 梯度累积 | 1 |
| 最大序列长度 | 1024 |
| Epoch | 1 |
| 总步数 | 1,073 |
| 初始学习率 | 4e-8 |
| DPO beta | 0.15 |
| 优化器 | AdamW |
| 混合精度 | bfloat16 |
| 梯度裁剪 | 1.0 |
| 每进程 DataLoader workers | 4 |
| 日志间隔 | 20 step |
| 保存间隔 | 500 step |
| `torch.compile` | 开启 |
| SwanLab | offline |

DPO 使用远低于 SFT 的学习率，是为了在学习偏好的同时降低灾难性遗忘风险。

## 9. 启动命令

工作目录为 `trainer/`：

```bash
cd /home/jk/work/minimind/trainer

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SWANLAB_MODE=offline \
PYTHONUNBUFFERED=1 \
../.venv/bin/torchrun --standalone --nproc_per_node=4 train_dpo.py \
  --data_path ../dataset/dpo.jsonl \
  --save_dir ../out \
  --save_weight dpo_full \
  --from_weight full_sft_full \
  --from_resume 0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 1024 \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --learning_rate 4e-8 \
  --beta 0.15 \
  --dtype bfloat16 \
  --num_workers 4 \
  --grad_clip 1.0 \
  --log_interval 20 \
  --save_interval 500 \
  --use_compile 1 \
  --use_wandb \
  --wandb_project MiniMind-DPO-Full-4GPU \
  >> ../logs/dpo_full_4gpu_20260927.log 2>&1
```

## 10. 本次训练结果

- 状态：已完成（2026-09-28 00:00，Asia/Shanghai）；
- 最终位置：Epoch 1，step 1073/1073；
- 初期 DPO loss 接近 $\ln 2\approx0.693$；
- 最终一个记录 batch 的 DPO loss：0.5616；
- `aux_loss=0`；
- 训练过程中没有 NaN、CUDA OOM、进程异常或过热；
- 训练结束后四张 GPU 均已释放。

DPO loss 会随不同偏好对的难度而波动，不要求每个 batch 单调下降。它从约 0.693 降到更低水平，表示策略模型整体上开始比初始参考模型更倾向 chosen 回答。但训练 loss 不能单独证明所有能力都提高，仍需独立偏好验证集和生成评测。

## 11. 日志与产物

```text
训练日志：       logs/dpo_full_4gpu_20260927.log
最终推理权重：   out/dpo_full_768.pth
完整恢复检查点： checkpoints/dpo_full_768_resume.pth
SwanLab 离线记录：trainer/swanlog/run-20260927_235632-bdivuzzpqk51fpkh1dnmm
```

- 推理权重约 132 MiB；
- 完整恢复检查点约 619 MiB；
- 最终权重已使用 `strict=True` 完成加载验证；
- 加载结果为 `missing_keys=0`、`unexpected_keys=0`。

SwanLab 离线记录可以同步到网页：

```bash
cd /home/jk/work/minimind
source .venv/bin/activate
swanlab login
swanlab sync trainer/swanlog/run-20260927_235632-bdivuzzpqk51fpkh1dnmm
```

## 12. DPO 与 SFT 的区别

| 项目 | SFT | DPO |
|---|---|---|
| 数据 | prompt + 标准回答 | prompt + chosen + rejected |
| 学习目标 | 提高标准回答 token 的概率 | 提高 chosen 相对 rejected 的偏好概率 |
| loss | token 多分类交叉熵 | 回答对二分类交叉熵 |
| 参考模型 | 不需要 | 需要冻结的 SFT 模型 |
| 在线生成 | 不需要 | 不需要 |
| Reward Model | 不需要 | 不需要显式 Reward Model |
| 更新模型 | 一个策略模型 | 只更新策略模型，参考模型冻结 |
| 计算开销 | 一套回答前向 | chosen/rejected × policy/reference |

DPO 不是继续模仿 chosen 的普通 SFT。它优化的是策略模型相对于参考模型的偏好 margin，同时既提高 chosen 的相对优势，也抑制 rejected 的相对优势。

## 13. 训练后的评测建议

1. 使用完全相同的 prompts 对比 SFT 与 DPO 模型输出。
2. 准备训练集之外的 chosen/rejected 验证对，统计 DPO 模型的偏好准确率。
3. 检查安全性、帮助性、事实性、重复程度和回答长度是否发生变化。
4. 同时保留 SFT 基准评测，确认 DPO 没有造成明显能力退化。
5. 不要只依据最后一个 batch 的 loss 判断模型质量，应观察平滑曲线和独立评测结果。

DPO 更适合进行偏好和安全对齐，并不等价于提升数学推理或知识能力。最终效果主要取决于偏好数据的质量、覆盖范围以及 chosen/rejected 是否真正体现目标偏好。
