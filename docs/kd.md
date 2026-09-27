# MiniMind 知识蒸馏原理与训练记录

## 1. 知识蒸馏是什么

知识蒸馏（Knowledge Distillation，KD）使用一个教师模型指导学生模型训练。目标不仅是让学生学习数据中的标准答案，还可以让学生拟合教师模型对整个词表的概率分布。

蒸馏通常分为两类：

- 黑盒蒸馏：只能看到教师生成的最终答案，再使用这些答案进行 SFT；
- 白盒蒸馏：可以访问教师 logits，让学生直接学习教师的 token 概率分布。

MiniMind 的 `trainer/train_distillation.py` 实现的是白盒蒸馏，训练目标由真实标签交叉熵和教师/学生分布之间的 KL 散度共同组成。

## 2. 黑盒蒸馏

黑盒蒸馏先让教师模型生成回答，再把生成结果作为硬标签训练学生：

$$
\mathcal L_{blackbox}=\mathrm{CE}(y_{teacher},p_{student})
$$

它的代码和普通 SFT 几乎相同。学生只能知道教师最终选择了哪个 token，无法看到教师对其他候选 token 的相对偏好。

MiniMind 的 SFT 数据中已经包含一部分由更强模型合成的 tool call、reasoning 和对话数据，因此广义上也包含黑盒蒸馏成分。

## 3. 白盒蒸馏

假设教师和学生对某个位置输出 logits：

$$
z_t,\quad z_s
$$

使用温度 $T$ 软化概率分布：

$$
p_t^T=\mathrm{softmax}(z_t/T)
$$

$$
p_s^T=\mathrm{softmax}(z_s/T)
$$

温度越高，概率分布越平滑，学生能看到教师对非最高概率 token 的相对判断。蒸馏损失为：

$$
\mathcal L_{KD}=T^2\mathrm{KL}(p_t^T\parallel p_s^T)
$$

乘以 $T^2$ 用于补偿温度缩放造成的梯度尺度变化。MiniMind 最终使用混合目标：

$$
\mathcal L=
\alpha\mathcal L_{CE}
+(1-\alpha)\mathcal L_{KD}
$$

其中：

- CE 让学生继续拟合真实 assistant 答案；
- KL 让学生的 token 分布接近教师；
- $\alpha$ 控制硬标签与软标签的权重；
- 教师模型冻结，只更新学生模型。

## 4. 本次教师与学生的选择

当前工作区没有更大尺寸或 MoE 教师权重，因此本次采用：

```text
教师：DPO 对齐模型 out/dpo_full_768.pth
学生：SFT 模型      out/full_sft_full_768.pth
```

两者都是 `hidden_size=768`、8 层、63.91M 参数的 Dense 模型。教师经过 DPO 偏好对齐，学生从 SFT 权重开始，通过 CE 和 KL 继续训练。

这次实验属于同容量模型之间的能力/偏好迁移，不是模型压缩：

- 学生参数量没有减小；
- 推理速度和模型文件大小不会因为本次 KD 自动下降；
- 教师与学生初始参数差异较小，所以初始 KL loss 很小；
- 训练效果仍需通过独立生成与偏好评测验证，不能只看训练 loss。

如果后续具备更大的 Dense 或 MoE 教师，可以保持学生为 63.91M 参数，用更强教师执行真正的“大模型蒸馏到小模型”。

## 5. 数据与标签

本次使用 README 默认的：

```text
dataset/sft_t2t_mini.jsonl
```

数据量为 905,718 条对话。数据通过 `SFTDataset` 处理：

1. 读取 `conversations`；
2. 应用 MiniMind Chat Template；
3. Tokenize 并截断或填充到 340 token；
4. system、user 和 padding 位置的 label 设置为 `-100`；
5. 只在 assistant 回答位置计算 CE 与 KL。

这样教师和学生读取完全相同的上下文，蒸馏信号只作用在需要模型生成的 assistant token 上。

## 6. 代码中的蒸馏流程

核心代码位于 `trainer/train_distillation.py`。

学生模型正常计算梯度：

```python
res = model(input_ids)
student_logits = res.logits[..., :-1, :]
```

教师模型冻结并关闭梯度：

```python
teacher_model.eval()
teacher_model.requires_grad_(False)

with torch.no_grad():
    teacher_logits = teacher_model(input_ids).logits[..., :-1, :]
```

真实标签 CE：

```python
ce_loss = F.cross_entropy(
    student_logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction="none"
)
```

教师/学生软分布 KL：

```python
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

distill_loss = temperature**2 * F.kl_div(
    student_log_probs,
    teacher_probs,
    reduction="batchmean"
)
```

混合损失：

```python
loss = alpha * ce_loss + (1 - alpha) * distill_loss
```

完整数据流为：

```text
JSONL 对话
  → Chat Template 与 Tokenize
  → assistant token mask
  → 学生模型前向传播（有梯度）
  → 教师模型前向传播（冻结、无梯度）
  → 真实标签 CE
  → 温度缩放后的 KL
  → 0.5 × CE + 0.5 × KL
  → 反向传播
  → 四卡 DDP 梯度同步
  → 梯度裁剪
  → AdamW 更新学生参数
```

## 7. 本次训练配置

| 配置 | 值 |
|---|---:|
| 数据集 | `sft_t2t_mini.jsonl` |
| 样本数 | 905,718 |
| 教师权重 | `dpo_full_768.pth` |
| 学生初始权重 | `full_sft_full_768.pth` |
| 教师参数量 | 63.912M |
| 学生参数量 | 63.912M |
| GPU 数量 | 4 |
| 每卡 batch size | 32 |
| 全局 batch size | 128 |
| 梯度累积 | 1 |
| 最大序列长度 | 340 |
| Epoch | 1 |
| 总步数 | 7,076 |
| 初始学习率 | 5e-6 |
| CE 权重 $\alpha$ | 0.5 |
| KL 权重 | 0.5 |
| 蒸馏温度 $T$ | 1.5 |
| 优化器 | AdamW |
| 混合精度 | bfloat16 |
| 梯度裁剪 | 1.0 |
| 每进程 DataLoader workers | 4 |
| 日志间隔 | 20 step |
| 保存间隔 | 500 step |
| `torch.compile` | 开启 |
| SwanLab | offline |

四卡训练日志中的总 loss、CE、aux loss 和 KL loss 会在所有 rank 之间求平均后再记录。

## 8. 启动命令

训练脚本需要在 `trainer/` 目录执行：

```bash
cd /home/jk/work/minimind/trainer

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SWANLAB_MODE=offline \
PYTHONUNBUFFERED=1 \
../.venv/bin/torchrun --standalone --nproc_per_node=4 train_distillation.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --save_dir ../out \
  --save_weight kd_dpo_teacher \
  --from_student_weight full_sft_full \
  --from_teacher_weight dpo_full \
  --from_resume 0 \
  --student_hidden_size 768 \
  --student_num_layers 8 \
  --teacher_hidden_size 768 \
  --teacher_num_layers 8 \
  --student_use_moe 0 \
  --teacher_use_moe 0 \
  --max_seq_len 340 \
  --epochs 1 \
  --batch_size 32 \
  --accumulation_steps 1 \
  --learning_rate 5e-6 \
  --alpha 0.5 \
  --temperature 1.5 \
  --dtype bfloat16 \
  --num_workers 4 \
  --grad_clip 1.0 \
  --log_interval 20 \
  --save_interval 500 \
  --use_compile 1 \
  --use_wandb \
  --wandb_project MiniMind-KD-DPO-Teacher-4GPU \
  >> ../logs/kd_dpo_teacher_4gpu_20260928.log 2>&1
```

## 9. 本次训练结果

- 状态：已完成（2026-09-28 00:46，Asia/Shanghai）；
- 最终位置：Epoch 1，step 7076/7076；
- 总耗时：约 39 分钟，包含数据解析、编译、检查点保存和早期温控暂停；
- 最终一个记录 batch 的总 loss：0.6432；
- 最终一个记录 batch 的 CE loss：1.2833；
- 最终一个记录 batch 的 KL loss：0.0031；
- Dense 学生模型 `aux_loss=0`；
- 没有出现 NaN、CUDA OOM、DDP 错误或进程崩溃。

初始 KL loss 约为 0.001，训练过程中通常在约 0.002～0.004 之间波动。原因是教师和学生结构相同，且 DPO 教师由同一个 SFT 模型继续训练而来，两者初始分布已经非常接近。随着 CE 更新学生参数，KL 项会限制学生过度偏离 DPO 教师。

最后一个 batch 的数值不能代表全数据平均值，也不能直接证明模型质量提升；应结合平滑曲线和独立验证集判断。

## 10. 日志与训练产物

```text
训练日志：       logs/kd_dpo_teacher_4gpu_20260928.log
最终学生权重：   out/kd_dpo_teacher_768.pth
完整恢复检查点： checkpoints/kd_dpo_teacher_768_resume.pth
SwanLab 离线记录：trainer/swanlog/run-20260928_000703-bdivuzzpqk51fpkh1dnmm
```

- 最终学生权重约 132 MiB；
- 完整恢复检查点约 619 MiB；
- 最终权重包含 91 个张量；
- 已使用 `strict=True` 完成加载验证；
- 验证结果为 `missing_keys=0`、`unexpected_keys=0`。

同步 SwanLab：

```bash
cd /home/jk/work/minimind
source .venv/bin/activate
swanlab login
swanlab sync trainer/swanlog/run-20260928_000703-bdivuzzpqk51fpkh1dnmm
```

## 11. 温控情况

训练初期系统服务 `gpu-temp-guard` 在 GPU 超过 80°C 时会暂停对应训练进程 60 秒。根据用户要求，本次训练中途执行了：

```bash
sudo systemctl stop gpu-temp-guard
```

随后四卡不再发生软件暂停，全速完成 KD。连续高负载时 GPU 温度达到约 88°C，并间歇出现 NVIDIA 硬件级热降频标志；训练进程和数值保持正常。

当前 `gpu-temp-guard` 状态为 `inactive`。这里只执行了 `stop`，没有执行 `disable`，因此机器重启后该服务仍会自动启动。如果要在不重启机器的情况下恢复软件温控，可执行：

```bash
sudo systemctl start gpu-temp-guard
```

关闭软件保护不会关闭 GPU 固件自身的热保护，但长时间高温可能降低稳定性和硬件寿命。

## 12. 如何恢复训练

如果需要从最近一次完整断点继续，将原命令改为：

```bash
--from_resume 1
```

脚本会从 `checkpoints/kd_dpo_teacher_768_resume.pth` 恢复学生模型、优化器、scaler、epoch 和 step。教师权重和模型结构必须与原训练保持一致。

## 13. 后续评测建议

应使用相同 prompts 至少比较以下三个模型：

```text
SFT： out/full_sft_full_768.pth
DPO： out/dpo_full_768.pth
KD：  out/kd_dpo_teacher_768.pth
```

建议评测：

1. 通用对话和指令遵循能力；
2. 独立偏好对上的 chosen/rejected 准确率；
3. 与 DPO 教师的 token 分布 KL；
4. 回答事实性、重复程度、长度和结束行为；
5. 原有任务是否发生灾难性遗忘。

只有当 KD 学生在独立评测中优于原 SFT 学生，或者更接近 DPO 教师且没有明显能力退化时，才能认为此次蒸馏产生了实际收益。
