# 使用 Kronos 对 贵州茅台（600519）进行微调

在量化交易领域，通用大模型往往需要针对特定标的进行微调（Fine-tuning），以捕捉个股独特的波动规律。本文将详细介绍如何使用 **Kronos** 模型对 000001 的 5 分钟 K 线数据进行微调，并展示训练过程与预测结果。

## 1. 数据准备

本次微调使用的是 CSV 格式的自定义数据。

- **数据路径**: `/home/luyangcai/code/Kronos/finetune_csv/data/HK_ali_09988_kline_5min_all.csv`
- **数据频度**: 5 分钟 (5min)
- **包含字段**: `timestamps` (时间戳), `open`, `high`, `low`, `close` (价格), `volume`, `amount` (成交量/额)

确保数据清洗干净且无缺失值，这是模型训练的基础。

## 2. 微调配置详解

我们使用 YAML 配置文件来管理所有超参数。

- **配置文件**: `/home/luyangcai/code/Kronos/finetune_csv/configs/config_ali09988_candle-5min.yaml`

### 关键参数解读

1.  **窗口设置**:
    ```yaml
    lookback_window: 512  # 模型"看"过去 512 个 5分钟K线 (约 5-6 个交易日)
    predict_window: 48    # 模型预测未来 48 个 5分钟K线 (4 小时，即半个交易日)
    ```

2.  **训练策略**:
    *   **Tokenizer**: 训练 30 轮 (`epochs: 30`)，学习率 `2e-4`。分词器需要适应个股的具体价格区间（如 60-100 HKD）。
    *   **Base Model**: 训练 20 轮 (`epochs: 20`)，学习率 `1e-6`。预测模型使用极小的学习率，在保留通用市场知识的同时，微调适应个股特性。

3.  **模型路径**:
    *   预训练模型: `NeoQuasar/Kronos-base`
    *   保存路径: `/home/luyangcai/code/Kronos/finetune_csv/finetuned/HK_ali_09988_kline_5min_all`

## 3. 训练过程与结果分析

训练日志位于: `/home/luyangcai/code/Kronos/finetune_csv/finetuned/HK_ali_09988_kline_5min_all/logs`

### 第一阶段：Tokenizer 微调

分词器的任务是将连续的价格序列离散化为 Token。
*   **初始 Loss**: ~0.0022
*   **最终 Loss**: ~0.0019 (趋于稳定)

**结论**: Loss 的下降表明 Tokenizer 能够以更高的精度重构阿里巴巴的历史价格数据，量化误差显著降低。

### 第二阶段：Base Model 微调

Base Model 负责学习 Token 序列的时序依赖关系。
*   **训练概况**: 共 20 轮。
*   **最佳表现**: 在第 4 轮 (Epoch 4) 达到最佳验证集损失 (**Validation Loss: 1.9831**)。

**结论**: 模型在几个 Epoch 内就迅速适应了新数据。由于我们设置了早停或保存最佳模型的机制，最终使用的是第 4 轮保存的 `best_model`，避免了后续可能的过拟合。

## 4. 使用微调模型进行预测

微调完成后，我们得到了一套专属的阿里巴巴预测模型。

### 加载模型

```python
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

# 1. 设置微调后的模型路径
finetuned_dir = "/home/luyangcai/code/Kronos/finetune_csv/finetuned/HK_ali_09988_kline_5min_all"
tokenizer_path = f"{finetuned_dir}/tokenizer/best_model"
model_path = f"{finetuned_dir}/basemodel/best_model"

# 2. 加载模型
print("正在加载微调后的模型...")
tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
model = Kronos.from_pretrained(model_path)

# 3. 初始化预测器
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
```

### 执行预测

```python
# 4. 准备输入数据 (取最近 512 个点)
df = pd.read_csv("/home/luyangcai/code/Kronos/finetune_csv/data/HK_ali_09988_kline_5min_all.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 512
pred_len = 48

x_df = df.iloc[-lookback:].copy()
x_timestamp = x_df['timestamps']
# 构造未来的时间戳 (假设)
y_timestamp = pd.date_range(start=x_timestamp.iloc[-1], periods=pred_len+1, freq='5min')[1:]

# 5. 预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    sample_count=1
)

print(pred_df.head())
```

通过以上步骤，我们成功将 Kronos 通用模型转化为阿里巴巴个股专用模型，为量化策略提供了更精准的 AI 信号支持。🚀