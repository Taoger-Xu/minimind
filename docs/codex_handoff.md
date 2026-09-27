# Codex 跨机器交接说明

更新时间：2026-09-28（Asia/Shanghai）

这份文件保存当前项目的工作上下文，供在另一台机器克隆仓库后交给 Codex 继续处理。它不包含原始数据集、模型权重、检查点、凭据或本机 Codex 会话缓存。

## 当前状态

- 预训练：已完成，说明见 `docs/ptrain.md`，日志与 loss 曲线位于 `logs/`。
- SFT：已完成，说明见 `docs/sft.md`，日志为 `logs/full_sft_4gpu_20260927.log`。
- DPO：已完成，说明见 `docs/dpo.md`，日志为 `logs/dpo_full_4gpu_20260927.log`。
- KD：已完成，说明见 `docs/kd.md`，日志为 `logs/kd_dpo_teacher_4gpu_20260928.log`。
- PPO：已实现并启动，随后按用户要求暂停；尚未完成，`docs/ppo.md` 也尚待最终整理。

## PPO 暂停点

- 日志：`logs/ppo_full_4gpu_20260928.log`
- 总步数：2438
- 最后记录：step 1070
- 最后指标：Reward -1.4943、KL_ref 0.0030、Approx KL 0.0002、ClipFrac 0、Critic Loss 0.1412、Avg Response Len 112.25
- 本机最新完整检查点：step 1000，`checkpoints/ppo_full_768_resume.pth`
- 本机策略权重：`out/ppo_full_768.pth`
- 当前机器上的 PPO 进程使用 `SIGSTOP` 暂停，不要在没有用户明确要求时恢复。

如果当前机器上的暂停进程仍存在，可用以下命令恢复：

```bash
pgrep -f '^/home/jk/work/minimind/.venv/bin/python .*train_ppo.py' | xargs -r kill -CONT
```

换机器后不能继承内存中的 step 1070，只能在准备好检查点和依赖后，从 step 1000 的检查点恢复。原始启动参数如下；恢复时把 `--from_resume 0` 改成 `--from_resume 1`：

```bash
cd trainer
env CUDA_VISIBLE_DEVICES=0,1,2,3 SWANLAB_MODE=offline PYTHONUNBUFFERED=1 \
../.venv/bin/torchrun --standalone --nproc_per_node=4 train_ppo.py \
  --data_path ../dataset/rlaif.jsonl \
  --save_dir ../out --save_weight ppo_full \
  --from_weight full_sft_full \
  --reward_model_path /home/jk/work/internlm2-1_8b-reward \
  --from_resume 1 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 \
  --max_seq_len 340 --max_gen_len 128 --epochs 1 --batch_size 2 \
  --mini_batch_size 2 --ppo_update_iters 2 --accumulation_steps 1 \
  --learning_rate 3e-7 --critic_learning_rate 5e-7 \
  --clip_epsilon 0.2 --cliprange_value 0.2 --vf_coef 0.5 \
  --kl_coef 0.02 --gamma 1.0 --lam 0.95 --early_stop_kl 0.25 \
  --thinking_ratio 0.9 --dtype bfloat16 --num_workers 2 \
  --grad_clip 1.0 --log_interval 10 --save_interval 250 \
  --rollout_engine torch --use_wandb \
  --wandb_project MiniMind-PPO-Full-4GPU \
  >> ../logs/ppo_full_4gpu_20260928.log 2>&1
```

## 未进入 Git 的文件

以下内容因体积、隐私或可复现性原因被排除，换机器继续训练前必须另行复制或重新下载：

- `dataset/*.jsonl`：所有训练数据，包括 `rlaif.jsonl`。
- `out/`：各阶段模型权重，PPO 还依赖 `out/full_sft_full_768.pth`。
- `checkpoints/`：优化器与训练恢复状态，PPO 恢复需要 `checkpoints/ppo_full_768_resume.pth`。
- `/home/jk/work/internlm2-1_8b-reward`：PPO 奖励模型，约 3.4 GB，位于仓库外。
- `.venv/`：本机 Python 虚拟环境。
- `trainer/swanlog/`：本机离线实验跟踪目录。

## 代码改动摘要

- 多卡预训练、SFT、DPO、KD、PPO 的日志指标改为跨 rank 聚合后由主进程记录。
- PPO 按 `log_interval` 记录指标，避免每一步产生冗余日志。
- 奖励模型 tokenizer 强制 `use_fast=False`，兼容 InternLM2 的 SentencePiece tokenizer。
- 新增预训练 loss 日志解析和绘图脚本 `scripts/plot_pretrain_loss.py`。

## 新机器继续方式

1. 克隆仓库并切换到本文件所在提交。
2. 对 Codex 说：`请先阅读 docs/codex_handoff.md 和相关训练文档，然后继续未完成的 PPO 工作。`
3. 另行准备上述数据集、SFT 权重、PPO checkpoint 和奖励模型。
4. 检查 CUDA、PyTorch、Transformers、SentencePiece 等环境后，再由用户明确决定是否恢复 PPO。

本机的 `gpu-temp-guard` 服务当前已停止但没有禁用；重启系统后它可能重新启用。
