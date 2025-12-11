### 1. 如何使用 Qlib 做训练？

Qlib 的训练流程主要分为三步：

1.  **准备数据**：使用 `finetune/qlib_data_preprocess.py` 从 Qlib 数据库中加载数据，并划分为训练集、验证集和测试集。
2.  **训练 Tokenizer**：使用 `finetune/train_tokenizer.py` 训练分词器，将连续的价格数据转化为离散的 Token。
3.  **训练 Predictor**：使用 `finetune/train_predictor.py` 训练预测模型，学习 Token 序列的规律并进行预测。

### 2. 训练的数据在哪里？

数据主要由两部分组成：
*   **原始数据**：通常存储在 `~/.qlib/qlib_data/cn_data`（Qlib 默认路径）。您需要在 `finetune/config.py` 中配置 `self.qlib_data_path` 指向该目录。
*   **处理后的数据**：运行预处理脚本后，生成的 `.pkl` 文件会保存在 `finetune/config.py` 中 `self.dataset_path` 指定的目录（默认是 `./data/processed_datasets`）。

### 3. 训练的方法与重要节点

我在代码中为您添加了以下备注，帮助您理解核心流程：

*   **配置 (`finetune/config.py`)**：这是“控制中心”。
    *   **关键点**：`qlib_data_path`（数据源）、`lookback_window`（看多长历史）、`predict_window`（预测多久未来）、`batch_size`（显存控制）、`learning_rate`（学习率）。
    *   *我在代码中添加了详细的参数说明。*

*   **数据预处理 (`finetune/qlib_data_preprocess.py`)**：这是“食材加工”。
    *   **关键点**：`load_qlib_data` 函数负责从 Qlib 读取数据，计算特征（如 Amount），并按时间切分。
    *   *我在代码中解释了数据加载和清洗的逻辑。*

*   **Tokenizer 训练 (`finetune/train_tokenizer.py`)**：这是“学习语言”。
    *   **关键点**：`train_model` 函数。它通过 VQ-VAE 架构，学习如何把价格变成 Token。Loss 包括 VQ Loss（量化误差）和 Reconstruction Loss（重构误差）。
    *   *我在代码中注释了训练循环和 Loss 的含义。*

*   **Predictor 训练 (`finetune/train_predictor.py`)**：这是“预测未来”。
    *   **关键点**：`train_model` 函数。它先用 Tokenizer 把数据变成 Token，然后像 GPT 一样训练 Predictor 预测下一个 Token。
    *   *我在代码中解释了 Tokenizer 编码和自回归训练的过程。*

### 下一步操作

请点击 **Review and Accept** 以应用这些中文注释。

应用后，如果您想开始训练（假设您已经有了 Qlib 数据）：

1.  **修改配置**：打开 `finetune/config.py`，确认 `qlib_data_path` 正确。
2.  **数据预处理**：
    ```bash
    python finetune/qlib_data_preprocess.py
    ```
3.  **训练 Tokenizer**：
    ```bash
    torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py # 假设有2张卡
    ```
4.  **训练 Predictor**：
    ```bash
    torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py
    ```