# 从数据到完整 LLM：MiniMind 项目全流程概览

本文面向已经了解 Transformer 基础架构、但尚未完整训练过大语言模型的读者。目标是结合 MiniMind 的代码，串起从数据收集、分词、模型构建，到损失计算、反向传播、参数更新、SFT 和偏好对齐的完整流程。

MiniMind 的核心训练逻辑可以概括为：

```text
收集原始文本
    ↓
清洗、去重、筛选并统一格式
    ↓
使用 Tokenizer 将文本转换为 Token ID
    ↓
构建 Decoder-Only Transformer
    ↓
预训练：Next-Token Prediction
    ↓
SFT：学习按照指令和对话格式回答
    ↓
DPO / RLAIF / Agentic RL：进一步进行偏好和任务对齐
    ↓
评估、推理与部署
```

> LLM 最基础的训练目标，是根据前面的 token 预测下一个 token。数据、Tokenizer、Transformer、损失函数、梯度和优化器都围绕这个目标组织。

## 1. 从一条真实样本看完整预训练过程

下面用一次从 `pretrain_t2t_mini.jsonl` 随机抽取的真实样本，先建立对整个预训练流程的直观认识。样本文本以如下内容开头：

```json
{"text": "In the Anareta 土星落陷，表示什么?在占星学中……"}
```

该文本共有 419 个字符。MiniMind 的预训练数据读取与处理逻辑位于 [`dataset/lm_dataset.py`](../dataset/lm_dataset.py)。

### 1.1 文本转换为固定长度序列

项目自带的 6400 词表 Tokenizer 将这条文本编码成 313 个正文 token，开头一部分 Token ID 为：

```text
[1486, 309, 4700, 1093, 4790, 256, ...]
```

然后添加 `BOS=1` 和 `EOS=2`：

```text
1 个 BOS + 313 个正文 token + 1 个 EOS
= 315 个有效 token
```

预训练默认 `max_seq_len=340`，因此再补充 25 个 `PAD=0`：

```text
input_ids： [BOS, 正文 token × 313, EOS, PAD × 25]
labels：    [BOS, 正文 token × 313, EOS, -100 × 25]

input_ids.shape = [340]
labels.shape    = [340]
```

`labels` 最初是 `input_ids` 的复制；PAD 位置被改为 `-100`，表示不参与损失计算。

### 1.2 DataLoader 组成 Batch

预训练默认一次读取 32 个样本。每条样本都被补齐或截断为 340 个位置，所以：

```text
单条样本：[340]
    ↓ 32 条样本堆叠
input_ids：[32, 340]
labels：   [32, 340]
```

其中 `32` 是 batch size，`340` 是每条样本的 token 序列长度。

### 1.3 模型前向传播

Token ID 先经过 Embedding，再经过 8 层 Decoder-Only Transformer，最后由 LM Head 映射到 6400 词表：

```text
input_ids                 [32, 340]
    ↓ Embedding
hidden_states             [32, 340, 768]
    ↓ 8 层 Transformer
hidden_states             [32, 340, 768]
    ↓ LM Head：768 → 6400
logits                    [32, 340, 6400]
```

`logits[b, t, :]` 表示第 `b` 条文本的第 `t` 个位置，对词表中全部 6400 个候选 token 的预测分数。因果掩码保证该位置只能根据自己和左侧上下文进行预测，不能看到未来 token。

### 1.4 Next-Token Prediction 与交叉熵

MiniMind 在模型内部将 logits 和 labels 错开一位：

```python
x = logits[..., :-1, :]  # [32, 339, 6400]
y = labels[..., 1:]      # [32, 339]
```

于是每个位置学习预测下一个 token：

```text
BOS              → 第一个正文 token
BOS, token 1     → token 2
...
全部正文          → EOS
```

对本次样本，错位后有 339 个位置，其中 314 个是有效预测目标，25 个 PAD 目标为 `-100`，会被忽略。整个 batch 在计算损失前展平为：

```text
logits：[32, 339, 6400] → [10848, 6400]
labels：[32, 339]       → [10848]
```

每个有效位置的交叉熵是正确 token 概率的负对数：

$$
L_t=-\log P(y_t\mid x_{\leq t})
$$

整个 batch 的损失是所有有效位置的平均：

$$
L
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log P(y_t\mid x_{\leq t})
$$

代码等价于：

```python
loss = F.cross_entropy(
    x.reshape(-1, 6400),
    y.reshape(-1),
    ignore_index=-100
)
```

如果随机初始化的模型对 6400 个 token 接近均匀预测，初始损失通常接近：

$$
-\log(1/6400)=\log 6400\approx8.76
$$

随着正确 token 的概率升高，损失逐渐下降。这就是 Next-Token Prediction，其概率模型为：

$$
P(x_1,x_2,\ldots,x_T)
=
\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1})
$$

### 1.5 反向传播与参数更新

预训练脚本随后执行：

```python
loss = loss / accumulation_steps
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

默认 `batch_size=32`、`accumulation_steps=8`，所以单卡约综合 $32\times8=256$ 条样本的梯度后更新一次参数。`backward()` 计算损失对 Embedding、Attention、MLP 和 LM Head 等全部参数的梯度；AdamW 根据梯度更新参数；更新完成后清空梯度。

这条流程会在整个数据集上重复：

```text
读取文本 → Tokenizer → Batch
→ Transformer 前向传播 → Next-Token Loss
→ 反向传播 → AdamW 更新参数
→ 正确 token 概率逐渐上升、Loss 逐渐下降
```

完整遍历一次数据集称为一个 epoch。大量重复后，随机初始化的模型逐渐学习词法、语法、上下文关系、知识和文本生成能力。后续章节将分别解释这条流水线中的数据、Tokenizer、模型、损失函数和训练阶段。

## 2. 数据收集与选择

### 2.1 数据决定模型能力

模型不会凭空获得知识。不同数据提供不同能力：

| 数据类型 | 主要能力 |
|---|---|
| 通用文本 | 语言规律、常识和文本续写 |
| 中文或英文数据 | 对应语言的理解与表达 |
| 代码数据 | 代码理解和生成 |
| 数学数据 | 计算和推理模式 |
| 多轮对话 | 对话和上下文跟随 |
| 指令数据 | 按用户要求完成任务 |
| 工具调用轨迹 | Tool Calling 和 Agent 能力 |
| 偏好数据 | 区分更好与更差的回答 |

模型架构决定容量和计算方式，数据决定模型实际学到的内容。

### 2.2 数据来源

常见来源包括百科、网页、书籍、论文、新闻、开源代码、公开问答、数学题、人工标注对话，以及教师模型合成或蒸馏的数据。

收集数据时必须同时考虑：

- 数据许可证和版权；
- 隐私与敏感信息；
- 有害或违法内容；
- 测试集污染；
- 数据来源是否允许传播和训练。

MiniMind 已经提供处理好的核心训练数据，因此复现项目时不需要重新处理大规模原始语料。

### 2.3 为什么原始数据不能直接训练

原始网页和文本通常包含广告、HTML、乱码、重复内容、无意义短句、内容农场文本和隐私信息。典型处理流程为：

```text
原始数据
  ↓
格式解析与正文抽取
  ↓
语言识别、乱码和 HTML 清理
  ↓
长度与质量过滤
  ↓
精确去重与模糊去重
  ↓
隐私及敏感信息处理
  ↓
质量打分与数据配比
  ↓
转成统一 JSONL 格式
```

### 2.4 如何选择数据

选择数据时重点关注：

1. **质量**：语句完整、信息密度高、逻辑连贯且内容正确。
2. **多样性**：覆盖不同语言、领域、长度、文体和任务。
3. **分布**：根据模型目标决定通用文本、代码、数学和对话等比例。
4. **重复率**：重复数据浪费算力并增加记忆和过拟合风险。
5. **难度与长度**：保留合理的长度和难度分布，避免样本过于单一。

对于 MiniMind 这样的小模型，少量高质量数据通常比大量低质量数据更有价值。

## 3. MiniMind 数据集与训练阶段

| 文件 | 训练阶段 | 作用 |
|---|---|---|
| `pretrain_t2t_mini.jsonl` | 轻量预训练 | 快速学习基础语言建模 |
| `pretrain_t2t.jsonl` | 主线预训练 | 更完整地学习语言和知识 |
| `sft_t2t_mini.jsonl` | 轻量 SFT | 快速获得 Zero 对话模型 |
| `sft_t2t.jsonl` | 主线 SFT | 学习对话、指令和 Tool Call |
| `dpo.jsonl` | DPO | 学习偏好回答 |
| `rlaif.jsonl` | PPO / GRPO / CISPO | 强化学习提示数据 |
| `agent_rl.jsonl` | Agentic RL | 多轮工具调用训练 |
| `agent_rl_math.jsonl` | RLVR | 带最终结果校验的数学推理 |

最小可用路线是：

```text
pretrain_t2t_mini.jsonl
          ↓
sft_t2t_mini.jsonl
          ↓
可对话的 MiniMind Zero
```

更完整的路线为：

```text
pretrain_t2t.jsonl
        ↓
sft_t2t.jsonl
        ├──→ DPO
        ├──→ RLAIF / GRPO / PPO / CISPO
        └──→ Agentic RL
```

DPO 和不同 RL 算法可以是从 SFT 模型出发的不同对齐路线，不一定要全部串行执行。

## 4. Tokenizer：从文本到数字

神经网络不能直接接收字符串，只能处理数字张量。Tokenizer 负责：

```text
自然语言 ⇄ token ⇄ Token ID
```

例如：

```text
文本：     人工智能正在发展
token：   [人工智能, 正在, 发展]
Token ID：[2941, 1678, 2253]
```

MiniMind 已经提供训练完成的 Tokenizer：

- [`model/tokenizer.json`](../model/tokenizer.json)
- [`model/tokenizer_config.json`](../model/tokenizer_config.json)
- 学习用训练脚本：[`trainer/train_tokenizer.py`](../trainer/train_tokenizer.py)

其主要配置为：

```text
算法：ByteLevel BPE
词表大小：6400
PAD / UNK ID：0
BOS ID：1
EOS ID：2
```

ByteLevel 保证几乎任意 Unicode 文本都能编码，BPE 则通过合并高频相邻符号，在词表大小和序列长度之间取得平衡。

预训练、SFT、DPO、RL 和推理必须使用同一个 Tokenizer。更换 Tokenizer 会改变 Token ID 的语义和切分方式，使原模型的 Embedding 与 LM Head 不再匹配。

## 5. 构建 Decoder-Only Transformer

MiniMind 的模型实现在 [`model/model_minimind.py`](../model/model_minimind.py)。主线结构是 Decoder-Only Transformer，默认核心配置包括：

```text
vocab_size：6400
hidden_size：768
num_hidden_layers：8
query heads：8
key/value heads：4
激活函数：SwiGLU
归一化：RMSNorm
位置编码：RoPE
max_position_embeddings：32768
默认：Dense
可选：MoE
```

整体数据流为：

```text
Token ID
   ↓
Token Embedding
   ↓
Transformer Block × 8
   ├── RMSNorm
   ├── Causal Self-Attention
   ├── Residual Connection
   ├── RMSNorm
   ├── SwiGLU MLP
   └── Residual Connection
   ↓
RMSNorm
   ↓
LM Head
   ↓
每个位置上 6400 个 token 的 logits
```

### 5.1 Embedding

Embedding 将离散 Token ID 映射为连续向量：

```python
nn.Embedding(vocab_size, hidden_size)
```

MiniMind 默认相当于：

```python
nn.Embedding(6400, 768)
```

每个 token 对应 Embedding 矩阵中的一行。训练开始时向量基本是随机的，训练后逐渐包含语义和使用规律。

### 5.2 自注意力

自注意力让当前位置从前文读取相关信息：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

可以粗略理解为：

- Query：当前位置想寻找什么；
- Key：前面各位置包含什么信息；
- Value：各位置实际提供的内容。

### 5.3 因果掩码

Next-Token Prediction 不能看到未来答案，所以 Decoder 使用 Causal Mask：

```text
        我  喜欢  人工  智能
我      ✓   ×    ×    ×
喜欢    ✓   ✓    ×    ×
人工    ✓   ✓    ✓    ×
智能    ✓   ✓    ✓    ✓
```

这使训练条件和推理条件一致：模型始终只能依据已有前文预测下一个 token。

### 5.4 MLP、RMSNorm、残差连接与 RoPE

- Attention 负责不同位置之间交换信息；
- MLP 对每个位置的信息进行非线性加工；
- 残差连接帮助信息和梯度穿过深层网络；
- RMSNorm 稳定网络各层的数值范围；
- RoPE 向注意力注入相对位置信息，使模型理解 token 顺序。

一个 Block 可以简化为：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

### 5.5 LM Head

Transformer 输出形状为：

```text
[batch_size, sequence_length, hidden_size]
```

LM Head 将每个位置的 768 维向量映射到 6400 个词表分数：

```python
nn.Linear(768, 6400)
```

输出形状变成：

```text
[batch_size, sequence_length, vocab_size]
```

这些未经归一化的分数称为 logits。

## 6. 数据如何进入模型

数据处理集中在 [`dataset/lm_dataset.py`](../dataset/lm_dataset.py)。

### 6.1 预训练数据

预训练样本为：

```json
{"text": "清晨的阳光透过窗帘洒进房间。"}
```

`PretrainDataset` 会：

1. 读取 `text`；
2. 使用 Tokenizer 编码；
3. 根据 `max_seq_len` 截断；
4. 添加 BOS 和 EOS；
5. 使用 PAD 补齐；
6. 复制 `input_ids` 得到 `labels`；
7. 把 PAD 对应的 label 设置为 `-100`。

例如：

```text
tokens：   [521, 1832, 2941]
加边界：   [1, 521, 1832, 2941, 2]
补齐：     [1, 521, 1832, 2941, 2, 0, 0]
labels：   [1, 521, 1832, 2941, 2, -100, -100]
```

`-100` 表示该位置不参与交叉熵计算。

### 6.2 SFT 数据

SFT 样本是结构化对话：

```json
{
  "conversations": [
    {"role": "user", "content": "什么是人工智能？"},
    {"role": "assistant", "content": "人工智能是……"}
  ]
}
```

Tokenizer 中的 Chat Template 首先把对话转成模型约定的文本：

```text
<|im_start|>user
什么是人工智能？<|im_end|>
<|im_start|>assistant
人工智能是……<|im_end|>
```

随后再编码为 Token ID。MiniMind 的 `SFTDataset` 只对 assistant 回答和结束标记计算损失：

```text
system 内容     → label = -100
user 内容       → label = -100
assistant 前缀  → label = -100
assistant 回答  → 正常 label
EOS             → 正常 label
padding         → label = -100
```

这样训练的目标就是：给定 system 和 user 内容，生成正确的 assistant 回答。

## 7. Next-Token Prediction 的前向计算

下面用一个只有 5 个 token 的小词表串起维度变化。假设词表为：

```text
0 = BOS，1 = 我，2 = 喜欢，3 = AI，4 = EOS
```

训练文本为 `[BOS, 我, 喜欢, AI, EOS]`。Next-Token Prediction 将输入与目标错开一位：

```text
input_ids = [[BOS, 我,   喜欢, AI ]]
labels    = [[我,  喜欢, AI,   EOS]]

input_ids.shape = [1, 4]
labels.shape    = [1, 4]
```

其中 `1` 是 batch size，`4` 是序列中的预测位置数。模型在每个位置都要给词表中的 5 个候选 token 打分，因此：

```text
input_ids                  logits
[1, 4]  → Transformer →  [1, 4, 5]
                             │  │  └─ 每个位置的 5 个候选 token
                             │  └──── 4 个预测位置
                             └─────── 1 个样本
```

`logits[0, 0, :]` 表示看到 `BOS` 后对 5 个候选 token 的评分，`logits[0, 1, :]` 表示看到 `BOS, 我` 后的评分，以此类推。模型需要完成：

| 位置 | 已知上下文 | 正确 label |
|---:|---|---|
| 0 | `BOS` | `我`（ID 1） |
| 1 | `BOS, 我` | `喜欢`（ID 2） |
| 2 | `BOS, 我, 喜欢` | `AI`（ID 3） |
| 3 | `BOS, 我, 喜欢, AI` | `EOS`（ID 4） |

Softmax 将 logits 转成概率：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

假设四个位置经过 Softmax 后，正确 label 的概率分别为：

```text
P(我    | BOS)          = 0.649
P(喜欢  | BOS, 我)      = 0.553
P(AI    | BOS, 我, 喜欢) = 0.503
P(EOS   | BOS, ..., AI) = 0.753
```

真实 MiniMind 的词表大小是 6400，因此实际输出例如：

```text
input_ids：[32, 340]
logits：   [32, 340, 6400]
```

即 32 个样本，每个样本 340 个预测位置，每个位置对 6400 个候选 token 打分。

## 8. 交叉熵损失

对于正确 token $y$，单个位置的交叉熵为：

$$
L=-\log p_y
$$

对上面四个位置，整个样本的平均损失为：

$$
\begin{aligned}
L
&= -\frac{1}{4}
[\log 0.649+\log 0.553+\log 0.503+\log 0.753] \\
&\approx 0.499
\end{aligned}
$$

正确 token 的概率越高，loss 越小；概率越低，loss 越大。一般形式是所有有效位置负对数概率的平均：

$$
L
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log P(x_{t+1}\mid x_{\leq t})
$$

PyTorch 的 `cross_entropy` 直接接收 logits，内部完成 LogSoftmax，不需要手动先算概率。计算前会合并 batch 和序列维度：

```python
# [1, 4, 5] → [4, 5]
flat_logits = logits.reshape(-1, vocab_size)

# [1, 4] → [4]
flat_labels = labels.reshape(-1)

loss = cross_entropy(
    flat_logits,
    flat_labels,
    ignore_index=-100
)

# loss.shape = []，结果是一个标量，例如 0.499
```

这里每一行 `[5]` logits 与一个正确 Token ID 比较。真实 MiniMind 中对应的是 `[6400]` logits 与一个正确 Token ID 比较。PAD 和 SFT 中不训练的位置会把 label 设为 `-100`，不参与平均损失。

## 9. 反向传播与梯度

将全部模型参数统一写作 $\theta$：

$$
\operatorname{logits}=f_\theta(x)
$$

$$
L(\theta)=\operatorname{CrossEntropy}(f_\theta(x),y)
$$

梯度表示每个参数发生微小变化时，损失如何变化：

$$
\nabla_\theta L=\frac{\partial L}{\partial \theta}
$$

PyTorch 在前向传播期间记录计算图，调用：

```python
loss.backward()
```

后会沿计算图反向应用链式法则：

```text
loss
  ↓
Softmax / Cross-Entropy
  ↓
LM Head
  ↓
Transformer Blocks
  ↓
Embedding
```

如果某个参数经过隐藏状态和 logits 影响损失，则：

$$
\frac{\partial L}{\partial W}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial h}
\frac{\partial h}{\partial W}
$$

开发者不需要手工计算数百万个梯度，PyTorch Autograd 会自动完成。

## 10. 优化器如何更新参数

最简单的梯度下降为：

$$
\theta_{new}
=
\theta_{old}-\eta\nabla_\theta L
$$

其中 $\eta$ 是学习率。沿梯度反方向移动参数，可以在局部降低损失。

MiniMind 使用 AdamW：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)
```

AdamW 会维护梯度的一阶、二阶滑动统计，并对参数应用权重衰减。一次最简训练迭代是：

```python
# 前向传播
result = model(input_ids, labels=labels)
loss = result.loss

# 反向传播
loss.backward()

# 参数更新
optimizer.step()

# 清空旧梯度
optimizer.zero_grad()
```

对应的 MiniMind 预训练脚本是 [`trainer/train_pretrain.py`](../trainer/train_pretrain.py)。

## 11. 训练中的工程机制

MiniMind 的实际训练循环还使用了混合精度、梯度累积和梯度裁剪：

```python
with autocast_ctx:
    result = model(input_ids, labels=labels)
    loss = result.loss / accumulation_steps

scaler.scale(loss).backward()

if step % accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

### 11.1 梯度累积

当显存只能容纳较小 batch 时，可以连续累积多个 micro batch 的梯度，再更新一次参数：

$$
\text{有效 batch size}
\approx
\text{batch size}
\times
\text{accumulation steps}
\times
\text{GPU 数量}
$$

### 11.2 混合精度

BF16 或 FP16 可以降低显存占用并提高 GPU 矩阵计算速度。MiniMind 默认使用 BF16；FP16 训练时使用 GradScaler 降低数值下溢风险。

### 11.3 梯度裁剪

当某一步梯度突然过大时，参数可能剧烈变化。`clip_grad_norm_` 将整体梯度范数限制在阈值内，减少训练发散风险。

### 11.4 学习率调度

学习率决定每次更新幅度。通常训练初期需要适当预热，中后期逐渐降低，使模型先快速学习、再稳定收敛。

### 11.5 Checkpoint 与 DDP

训练脚本支持：

- 保存模型、优化器、Scaler 和训练进度；
- 使用 `--from_resume 1` 恢复训练；
- 使用 `torchrun` 启动 DDP 多卡训练；
- 使用 WandB 或 SwanLab 记录训练曲线。

## 12. 预训练：从随机参数学习语言

运行：

```bash
cd trainer
python train_pretrain.py
```

默认主要配置为：

```text
数据：../dataset/pretrain_t2t_mini.jsonl
hidden_size：768
层数：8
max_seq_len：340
batch_size：32
accumulation_steps：8
学习率：5e-4
epoch：2
初始权重：none
优化器：AdamW
```

训练前，Embedding、Attention、MLP 和 LM Head 参数基本是随机的。经过大量 Next-Token Prediction，模型逐步学习：

- 字词和语法规律；
- 上下文依赖；
- 基础知识；
- 文本风格；
- 连贯续写能力；
- 部分推理模式。

默认输出为：

```text
out/pretrain_768.pth
```

预训练模型主要擅长续写，不一定能稳定按照用户指令进行对话。

## 13. SFT：从续写模型到对话模型

运行：

```bash
cd trainer
python train_full_sft.py
```

实现位于 [`trainer/train_full_sft.py`](../trainer/train_full_sft.py)，默认主要配置为：

```text
数据：../dataset/sft_t2t_mini.jsonl
初始权重：pretrain
max_seq_len：768
batch_size：16
学习率：1e-5
epoch：2
```

SFT 仍然使用 Next-Token Prediction 和交叉熵。区别在于：

1. 数据是结构化对话；
2. 使用 Chat Template；
3. 只对 assistant 回答计算损失；
4. 从预训练权重继续训练；
5. 学习率通常显著低于预训练。

```text
预训练：学习语言、知识和续写
SFT：学习如何根据 system 和 user 指令生成 assistant 回答
```

默认输出为：

```text
out/full_sft_768.pth
```

完成预训练和 SFT 后，就得到了最基础的完整对话 LLM。

## 14. DPO：学习回答偏好

SFT 告诉模型一个问题可以怎样回答，DPO 则告诉模型两个回答中哪一个更好。

DPO 数据包含 chosen 和 rejected：

```json
{
  "chosen": [
    {"role": "user", "content": "如何学习 Python？"},
    {"role": "assistant", "content": "建议从语法、练习和项目开始……"}
  ],
  "rejected": [
    {"role": "user", "content": "如何学习 Python？"},
    {"role": "assistant", "content": "随便看看就行。"}
  ]
}
```

DPO 希望提高 chosen 相对 rejected 的概率，同时限制训练模型不要过度偏离参考模型：

$$
L_{DPO}
=
-\log\sigma\left(
\beta
\left[
\log\frac{\pi(y_w|x)}{\pi(y_l|x)}
-
\log\frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
$$

其中：

- $\pi$ 是正在训练的 policy model；
- $\pi_{ref}$ 是冻结的 reference model；
- $y_w$ 是 chosen；
- $y_l$ 是 rejected；
- $\beta$ 控制偏好优化强度。

代码位于 [`trainer/train_dpo.py`](../trainer/train_dpo.py)，默认从 `full_sft` 权重开始：

```bash
cd trainer
python train_dpo.py
```

## 15. RLAIF、GRPO、PPO 与 Agentic RL

强化学习阶段的共同框架是：

```text
给模型一个问题
    ↓
模型生成一个或多个回答（Rollout）
    ↓
奖励函数或奖励模型评分
    ↓
根据 Reward / Advantage 调整回答概率
```

关键概念包括：

- **Policy**：当前语言模型；
- **Rollout**：模型实际生成的轨迹；
- **Reward**：回答得到的评分；
- **Advantage**：回答相对基线好多少；
- **KL**：限制新模型不要偏离原模型过远；
- **Value Model**：在 PPO 中预测预期回报。

GRPO 会为同一问题生成多个回答，在组内比较奖励：

```text
问题
 ├── 回答 A → reward 0.9 → 正 advantage
 ├── 回答 B → reward 0.3 → 负 advantage
 ├── 回答 C → reward 0.6 → 略正 advantage
 └── 回答 D → reward 0.1 → 明显负 advantage
```

随后提高高奖励回答的生成概率，降低低奖励回答的生成概率。

Agentic RL 则在生成过程中加入工具交互：

```text
用户问题
   ↓
模型生成 <tool_call>
   ↓
执行工具
   ↓
返回 <tool_response>
   ↓
模型继续思考或回答
   ↓
根据任务结果计算奖励
```

相关实现包括：

- [`trainer/train_grpo.py`](../trainer/train_grpo.py)
- [`trainer/train_ppo.py`](../trainer/train_ppo.py)
- [`trainer/train_agent.py`](../trainer/train_agent.py)
- [`trainer/rollout_engine.py`](../trainer/rollout_engine.py)

对于初学者，应先完全理解预训练和 SFT，再学习 DPO 和强化学习。

## 16. 完整训练循环

预训练和 SFT 的核心可以浓缩为：

```python
model = MiniMindForCausalLM(config)
optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for input_ids, labels in dataloader:
        # 1. 前向传播
        outputs = model(input_ids)

        # 2. 当前位置预测下一个 token
        shift_logits = outputs.logits[:, :-1]
        shift_labels = labels[:, 1:]

        # 3. 交叉熵损失
        loss = cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=-100
        )

        # 4. 自动微分
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        # 6. 清空梯度
        optimizer.zero_grad()
```

所有工程机制都围绕这六步扩展：

- DDP：多 GPU 并行；
- BF16 / FP16：减少显存并加速；
- 梯度累积：模拟更大 batch；
- 梯度裁剪：降低训练发散风险；
- 学习率调度：控制不同阶段的更新幅度；
- Checkpoint：支持暂停和恢复；
- WandB / SwanLab：记录训练曲线。

## 17. 如何判断训练是否正常

### 17.1 Loss

```text
loss 长期下降      → 模型通常正在学习
loss 完全不变      → 参数可能没有正确更新
loss 变成 NaN      → 数值不稳定、数据或学习率可能异常
loss 剧烈震荡      → 学习率、batch 或数据分布可能有问题
训练 loss 很低但验证差 → 可能过拟合
```

不要只观察单个 step，应关注平滑后的长期趋势。

### 17.2 Perplexity

困惑度定义为：

$$
PPL=e^{Loss}
$$

它可以粗略表示模型对下一个 token 有多不确定。但 PPL 会受到 Tokenizer 影响，只适合在相同 Tokenizer 和评估数据上直接比较。

### 17.3 训练集、验证集和测试集

```text
训练集：计算梯度并更新参数
验证集：只做前向计算，用于选择模型和超参数
测试集：在开发结束后进行最终评估
```

如果训练 loss 继续下降而验证 loss 上升，通常意味着过拟合。

### 17.4 下游能力

除了 loss，还应评估：

- 中英文问答；
- 文本续写；
- 数学和代码；
- 指令遵循；
- Tool Call 格式和成功率；
- 重复生成与幻觉；
- 长上下文表现。

## 18. 推荐实践路线

### 18.1 初学者快速复现

```bash
cd trainer

python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl

python train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl
```

然后回到项目根目录测试：

```bash
python eval_llm.py --weight full_sft
```

这条路线最适合理解：

```text
随机初始化
  ↓
预训练续写模型
  ↓
SFT 对话模型
```

### 18.2 完整主线数据

```bash
cd trainer

python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t.jsonl \
  --max_seq_len 380

python train_full_sft.py \
  --data_path ../dataset/sft_t2t.jsonl
```

完整数据训练成本明显高于 mini 数据。

### 18.3 偏好对齐

SFT 完成后，可以选择 DPO：

```bash
cd trainer

python train_dpo.py \
  --data_path ../dataset/dpo.jsonl \
  --from_weight full_sft
```

也可以根据目标选择 GRPO、PPO、CISPO 或 Agentic RL。初学阶段不建议一开始就把所有训练阶段串在一起。

## 19. 推荐阅读代码的顺序

1. **Tokenizer**：理解文本如何转换为 Token ID。
   - [`trainer/train_tokenizer.py`](../trainer/train_tokenizer.py)
   - [`model/tokenizer_config.json`](../model/tokenizer_config.json)
2. **数据集**：理解 `input_ids`、`labels` 和 loss mask。
   - [`dataset/lm_dataset.py`](../dataset/lm_dataset.py)
3. **模型**：理解 Embedding、Attention、MLP 和 LM Head。
   - [`model/model_minimind.py`](../model/model_minimind.py)
4. **预训练**：理解 forward、loss、backward 和 optimizer。
   - [`trainer/train_pretrain.py`](../trainer/train_pretrain.py)
5. **SFT**：理解 Chat Template 和 assistant-only loss。
   - [`trainer/train_full_sft.py`](../trainer/train_full_sft.py)
6. **推理**：理解逐 token 生成。
   - [`eval_llm.py`](../eval_llm.py)
7. **DPO**：理解偏好学习。
   - [`trainer/train_dpo.py`](../trainer/train_dpo.py)
8. **强化学习**：最后学习 PPO、GRPO、CISPO 和 Agentic RL。

## 20. 总结

MiniMind 首先使用 ByteLevel BPE Tokenizer 将文本转换成 6400 词表内的 Token ID；Decoder-Only Transformer 使用因果注意力读取当前位置之前的上下文，在每个位置输出下一个 token 的概率分布；交叉熵衡量预测与真实 token 的差距；PyTorch 通过反向传播计算损失对每个参数的梯度；AdamW 再沿降低损失的方向更新参数。

不同阶段的目标是：

```text
预训练：学习语言、知识和续写
SFT：学习按照指令和对话格式回答
DPO：学习 chosen 优于 rejected 的偏好
RL：通过奖励进一步优化推理、任务和工具调用能力
```

最重要的闭环是：

```text
数据决定模型学习什么
Tokenizer 决定文本如何表示
Transformer 根据前文计算下一个 token 的概率
Loss 衡量模型预测有多错
Gradient 指出参数应如何变化
Optimizer 更新参数
重复大量步骤后，模型逐渐获得语言能力
```
