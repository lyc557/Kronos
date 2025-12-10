# 【实战】手把手教你微调 Kronos 量化大模型：从 CSV 到股价预测

在量化交易领域，如何利用深度学习模型精准捕捉市场趋势一直是热门话题。今天，我们将深入探讨一个强大的时间序列预测模型——**Kronos**，并手把手教你如何使用自己的股票数据（CSV格式）对其进行微调（Fine-tuning）。

无论你是量化新手还是 AI 爱好者，这篇文章都将带你跑通从数据准备到模型训练的全流程。

---

## 什么是 Kronos？

Kronos 是一个专为时间序列设计的生成式模型。与传统的 LSTM 或 Transformer 不同，Kronos 引入了类似 NLP（自然语言处理）中的 **Tokenizer** 概念：

1.  **Tokenizer (分词器)**：它不直接处理原始价格数值，而是先将连续的时间序列数据“翻译”成离散的 tokens（类似把句子切分成词）。这利用了 **VQ-VAE**（向量量化变分自编码器）技术。
2.  **Predictor (预测器)**：基于这些 tokens，模型学习历史序列的规律，并预测未来的 token 序列，最后再“翻译”回价格数值。

这种架构让 Kronos 能够捕捉到更复杂的市场模式。

---

## 第一步：准备你的数据

Kronos 的微调非常灵活，支持标准的 CSV 格式。你只需要准备好你的 K 线数据。

### 数据格式要求
你的 CSV 文件需要包含以下表头：
*   `timestamps`: 时间戳 (例如 `2024/01/01 09:30`)
*   `open`, `high`, `low`, `close`: 开高低收价格
*   `volume`: 成交量
*   `amount`: 成交额 (可选，没有填0即可)

**示例数据 (`data/HK_ali_09988_kline_5min_all.csv`)：**
```csv
timestamps,open,close,high,low,volume,amount
2019/11/26 9:35,182.45,184.45,184.95,182.45,15136000,0
2019/11/26 9:40,184.35,183.85,184.55,183.45,4433300,0
...
```

---

## 第二步：配置你的训练参数

Kronos 提供了一个强大的配置文件系统，让你无需修改代码即可调整参数。
打开 `configs/config_ali09988_candle-5min.yaml`，关键配置如下：

```yaml
data:
  # 你的数据路径
  data_path: "/path/to/your/data.csv"
  lookback_window: 512    # 历史回顾窗口（模型看多长的时间）
  predict_window: 48      # 预测窗口（模型预测多远的未来）

training:
  tokenizer_epochs: 30    # Tokenizer 训练轮数
  basemodel_epochs: 20    # Predictor 训练轮数
  batch_size: 32          # 批次大小
  
  # 【性能优化小贴士】
  # 如果你的训练出现 CPU 100% 负载，请调小这个参数（如改为 2 或 4）
  num_workers: 6          
```

---

## 第三步：一键启动微调

我们推荐使用 **顺序训练 (Sequential Training)** 模式。这个脚本会自动先训练 Tokenizer，再训练 Predictor，不仅省心，效果也更稳定。

在终端运行：
```bash
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml
```

如果你有多张显卡，还可以开启 DDP 分布式训练加速：
```bash
DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=8 train_sequential.py ...
```

---

## 第四步：读懂训练日志（关键！）

训练开始后，你会看到一串串神秘的数字。别慌，我们来解码这些关键指标：

### 1. Tokenizer 训练阶段
*   **VQ Loss (Vector Quantization Loss)**: 
    *   *现象*：这个值可能是**负数**。
    *   *解释*：别担心，这是正常的！因为它包含了一个“熵惩罚”项，用来鼓励模型使用更多样化的词表（Codebook）。只要它在波动中趋于稳定就是好现象。
*   **Recon Loss (Reconstruction Loss)**: 
    *   *解释*：重构损失。代表模型把价格转成 Token 再转回价格后的误差。
    *   *目标*：这个值越低越好，说明 Tokenizer 没“丢失”太多信息。

### 2. Base Model 训练阶段
*   **Training / Validation Loss**:
    *   *解释*：预测误差。
    *   *目标*：随着 epoch 增加，Loss 应该稳步下降。

---

## 结语

通过以上步骤，你就拥有了一个在特定标的（如阿里巴巴股票）上微调过的专属 Kronos 模型。

**下一步做什么？**
*   尝试不同的 `lookback_window`，看看长短期记忆对预测的影响。
*   使用 `examples/` 目录下的脚本可视化预测结果，直观感受 AI 的判断力。

量化交易的未来属于 AI，而 Kronos 正是你手中的一把利剑。Happy Trading! 🚀