# MiniMind 预训练记录

## 当前状态

- 状态：已完成（2026-09-27 05:09，Asia/Shanghai）
- 数据集：`dataset/pretrain_t2t.jsonl`，约 7.8 GiB、8,468,827 条样本
- 模型：63.91M 参数，`hidden_size=768`，8 层 Transformer，非 MoE
- 设备：4 × NVIDIA A100-SXM4-80GB
- 恢复来源：单卡 epoch 1、step 28500 检查点
- 四卡恢复位置：epoch 1、step 7125/66163
- 最终结果：epoch 2、step 66163/66163，loss 1.5467

单卡 step 会在恢复时按 GPU 数换算：`28500 ÷ 4 = 7125`。保持每卡
`batch_size=32`，可以确保切换前后已经消费的数据位置一致。

## 训练配置

| 配置 | 值 |
|---|---:|
| GPU 数量 | 4 |
| 每卡 batch size | 32 |
| 梯度累积 | 2 |
| 全局有效 batch size | 256 |
| 序列长度 | 380 |
| Epoch | 2 |
| 初始学习率 | 5e-4 |
| 优化器 | AdamW |
| 混合精度 | bfloat16 |
| 梯度裁剪 | 1.0 |
| 日志间隔 | 20 step |
| 保存间隔 | 5000 step |
| `torch.compile` | 开启 |
| SwanLab | offline |

全局有效 batch size：

```text
32（每卡）× 4（GPU）× 2（梯度累积）= 256
```

## 启动与恢复

工作目录必须是 `trainer/`：

```bash
cd /home/jk/work/minimind/trainer

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SWANLAB_MODE=offline \
PYTHONUNBUFFERED=1 \
../.venv/bin/torchrun --standalone --nproc_per_node=4 train_pretrain.py \
  --data_path ../dataset/pretrain_t2t.jsonl \
  --save_dir ../out \
  --save_weight pretrain_full_gpu0 \
  --from_weight none \
  --from_resume 1 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 380 \
  --epochs 2 \
  --batch_size 32 \
  --accumulation_steps 2 \
  --learning_rate 5e-4 \
  --dtype bfloat16 \
  --num_workers 4 \
  --grad_clip 1.0 \
  --log_interval 20 \
  --save_interval 5000 \
  --use_compile 1 \
  --use_wandb \
  --wandb_project MiniMind-Pretrain-Full-4GPU \
  >> ../logs/pretrain_full_4gpu_20260926.log 2>&1
```

`--from_resume 1` 会恢复模型、AdamW、混合精度、epoch 和 step；不要用
`--from_weight` 代替完整断点恢复。

## 数据与训练流程

1. `PretrainDataset` 从 JSONL 的 `text` 字段读取文本。
2. Tokenizer 添加 BOS/EOS，并截断或填充到 380 token。
3. `labels` 复制自 `input_ids`，PAD 位置改为 `-100`。
4. 每张 GPU 得到 `[32, 380]` 的 `input_ids` 和 `labels`。
5. 模型输出 `[32, 380, 6400]` 的 logits。
6. `logits[..., :-1, :]` 与 `labels[..., 1:]` 对齐，计算 next-token 交叉熵。
7. 每两个 micro-batch 累积一次梯度，随后裁剪梯度并执行 AdamW 更新。
8. DDP 在四张 GPU 间同步梯度；日志 loss 是四卡平均值。

## 日志与产物

```text
当前日志：       logs/pretrain_full_4gpu_20260926.log
历史单卡日志：   logs/pretrain_full_gpu0_resume_20260926.log
推理权重：       out/pretrain_full_gpu0_768.pth
完整恢复检查点： checkpoints/pretrain_full_gpu0_768_resume.pth
最终 Loss 曲线： logs/pretrain_full_loss_final.png
最终 Loss 数据： logs/pretrain_full_loss_final.csv
```

最终推理权重和完整恢复检查点均已保存。

## 查看与停止

```bash
# 实时日志
tail -f /home/jk/work/minimind/logs/pretrain_full_4gpu_20260926.log

# 四卡状态
nvidia-smi

# 正常停止 torchrun 及其训练进程
pkill -SIGINT -f 'torchrun.*train_pretrain.py'
```

再次执行上面的启动命令即可从最近一次完整检查点继续。

最终权重仍应在投入后续 SFT 前执行一次加载检查和简单文本生成测试。
