---
title: LLM Plan
---

**“大模型系统学习的 8 周详尽实操练习方案”**

整套方案覆盖：**从零手撸 GPT → 训练 → 微调 → RLHF → 推理部署 → 工程化 →论文阅读 → 多模态扩展**。
每周都有明确任务 + 推荐资料 + 验收成果，可直接当作学习路线图。

---

# 🧪 **8 周大模型实操强化计划（详细到每天能做什么）**

---

# ✅ **第 1 周：Transformer / GPT 基础实现（代码从零开始）**

目标：理解 Attention、位置编码、Transformer Block 的实质。
成果：自己写出一个最小可运行 Transformer。

### 📌 实操任务

1. **从零实现 Self-Attention**

   * 算法：Scaled Dot-Product Attention
   * 代码任务：

     * 自己实现 Q/K/V 线性层
     * 手写 attention scores
     * 做 softmax、加 mask、加 dropout

2. **手写位置编码 Positional Encoding（sin-cos）**

3. **搭建一个完整 TransformerBlock**

   * MultiHeadAttention
   * Feed Forward Network
   * LayerNorm + Residual

4. **训练一个 Mini GPT**

   * 训练字符级语言模型（如莎士比亚英文）
   * 验证能生成连贯文本

### 🔧 参考代码模型

* Karpathy *nanoGPT*
* The Annotated Transformer

### 🎯 本周验收成果

* 能运行 `model.generate()` 输出文本
* **100% 看懂 nanoGPT 代码**

---

# ✅ **第 2 周：GPT 训练与调参（从数据到完整训练管线）**

目标：掌握预训练数据准备、训练超参设置、loss 收敛逻辑。

### 📌 实操任务

1. **从零构建预训练数据集**

   * 使用 `tokenizers` 自己训练一个 BPE tokenizer（1~5k vocab）
   * 准备自己的语料（中英皆可）

2. **实现训练脚本**

   * batch sampler
   * AdamW
   * 学习率 warmup + cosine decay
   * gradient clipping

3. **跑一次预训练**

   * 训练一个 10M 参数的 GPT
   * 在自己的语料上 fine-tune

4. **分析模型表现**

   * loss 曲线
   * perplexity
   * 生成效果

### 🎯 本周验收成果

* 一个完整的 `train.py`
* 一个真实可用的小 GPT 模型

---

# ✅ **第 3 周：大模型工程化（并行训练、显存优化、速度优化）**

目标：理解工业级训练系统如何让 GPT-2/3 级模型跑起来。

### 📌 实操任务

1. **掌握高效训练框架**

   * DeepSpeed（ZeRO-1~3）
   * Megatron-LM（张量并行 TP）

2. **显存优化**

   * Gradient Checkpointing
   * FlashAttention
   * 模型并行
   * 混合精度训练（fp16/bf16）

3. **训练一个大点的模型**

   * GPT 100M ~ 300M
   * 使用 FlashAttention + Deepspeed

### 🎯 本周验收成果

* 能训练一个 1 亿参数级 GPT
* 学会阅读 DeepSpeed/Megatron 的日志与 profiler

---

# ✅ **第 4 周：微调方法（LoRA / QLoRA 全面掌握）**

目标：能对任何 LLaMA/Mistral/SFT 模型进行训练和部署。

### 📌 实操任务

1. **复现 LoRA**

   * 自己实现 LoRA 线性层
   * 对小模型进行 LoRA 微调

2. **使用 PEFT 库进行实际微调**

   * SFT
   * LoRA
   * QLoRA（4bit 量化）

3. **构建一个文本生成微调任务**

   * 小说续写
   * 编程助手
   * 食品识别文本生成（与你的项目相关）

4. **全参数 vs LoRA vs QLoRA 对比实验**

   * 参数量
   * 显存占用
   * 速度
   * 生成质量

### 🎯 本周验收成果

* 能微调 LLaMA-3 8B、Mistral 7B
* 学会用 QLoRA 在 1 张 3090 上训练

---

# ✅ **第 5 周：RLHF（PPO、DPO、RLAIF 全流程）**

目标：理解大模型“变聪明”的关键步骤，亲手复现 InstructGPT 过程。

### 📌 实操任务

1. **复现奖励模型（Reward Model）训练**
2. **使用 TRL 库跑 PPO 训练**
3. **使用 DPO 实现对齐**
4. **用 RLAIF（AI 自监督）代替人工数据**

### 实战例子：

* 搭建一个“你的个人小说风格助手”
* 搭建一个“多步推理助手”
* 构建一个“食品场景推断助手”（结合你的图像项目）

### 🎯 本周验收成果

* 训练自己的 RLHF 模型
* PPO + DPO pipeline 可运行

---

# ✅ **第 6 周：推理与部署（vLLM / TensorRT-LLM）**

目标：理解生产环境下如何让 LLM 高速运行、低成本推理。

### 📌 实操任务

1. **使用 vLLM 部署一个 7B LLaMA-3**

   * KV-cache 管理
   * 异步批处理
   * speculative decoding（推测式解码）

2. **使用 TensorRT-LLM 加速推理**

   * fp8 量化
   * CUDA Graphs
   * build engine

3. **搭建一个 Web API 服务**

   * FastAPI
   * OpenAI-compatible API

### 🎯 本周验收成果

* 本机/服务器可运行稳定的 LLM 接口
* Qwen/Mistral/LLaMA 任意模型都能部署

---

# ✅ **第 7 周：模型评测、调参、Prompt 结构化体系**

目标：掌握如何判断一个模型是否“好”、如何让它“更好”。

### 📌 实操任务

1. **自动化评测**

   * perplexity
   * BLEU / ROUGE
   * MMLU
   * Arena 评测（对齐度）

2. **Prompting 结构化工程**

   * ReAct
   * Tree of Thoughts
   * System Prompt 工程
   * 长文写作 Prompt 模块化（与你小说写作工具相关）

3. **输出风格控制实验**

   * memory
   * narrative style
   * persona control

### 🎯 本周验收成果

* 完成一份“模型评估报告”
* 设计一套 Prompt 模块化系统（可用于你的小说 App）

---

# ✅ **第 8 周：多模态与进阶研究（图像/语音/视频）**

结合你自己的研究方向（CV、图像生成项目、食物识别）
重点学习 LMM（Large Multimodal Models）：

### 📌 实操任务

1. **复现 CLIP（对标你的食物识别目标）**
2. **跑一次 BLIP / BLIP-2 Zero-shot Caption**
3. **用 Llava 或 Qwen-VL 做多模态微调**
4. **构建一个视觉推理管线**

   * 物体 → 属性 → 场景 → 消费推理
   * 与你的 food labeller 项目结合

### 🎯 本周验收成果

* 你自己的小型视觉语言模型
* 能对任意图像区域做解释

---

# 🌟 **额外 Bonus（适合你）**

因为你有数学背景 + 在做视频生成/3D 动画：

你可以在额外时间学习：

### 1️⃣ 低秩训练技巧

* LoRA 理论
* SVD 分解
* 低秩优化（与你的研究兴趣强关联）

### 2️⃣ Diffusion 源码阅读

适合你动画项目的实现。

### 3️⃣ 3D 生成（你的动画 pipeline 必学）

* LGM
* Gaussian Splatting
* 3D Diffusion

