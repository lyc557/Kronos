import os

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters (数据与特征参数)
        # =================================================================
        # TODO: Update this path to your Qlib data directory.
        # Qlib数据存储路径：这里存放了从Qlib下载的原始金融数据（如股票日线数据）
        # 默认路径通常是 ~/.qlib/qlib_data/cn_data
        self.qlib_data_path = "/home/ubuntu/.qlib/qlib_data/cn_data"
        # 股票池：指定要训练的股票集合，如 'csi300' (沪深300), 'csi500' (中证500) 或 'all' (全市场)
        self.instrument = 'csi300'

        # Overall time range for data loading from Qlib.
        # 数据加载总时间范围：从Qlib加载数据的起止时间
        self.dataset_begin_time = "2011-01-01"
        self.dataset_end_time = '2025-06-05'

        # Sliding window parameters for creating samples.
        # 滑动窗口参数：决定了如何构建训练样本
        self.lookback_window = 90  # 回顾窗口：模型根据过去90天的数据进行预测
        self.predict_window = 10  # 预测窗口：模型预测未来10天的走势
        self.max_context = 512  # 最大上下文：模型能处理的最大序列长度

        # Features to be used from the raw data.
        # 特征列表：输入模型的具体特征，包括开高低收、成交量、成交额
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # Time-based features to be generated.
        # 时间特征：辅助模型理解时间周期性（如星期几、几点钟）
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths (数据集划分与路径)
        # =================================================================
        # Note: The validation/test set starts earlier than the training/validation set ends
        # to account for the `lookback_window`.
        # 数据集划分：将时间轴划分为训练集、验证集、测试集、回测集
        # 注意：为了防止未来信息泄露，必须严格按时间顺序切分
        self.train_time_range = ["2011-01-01", "2022-12-31"]  # 训练集：用于模型学习
        self.val_time_range = ["2022-09-01", "2024-06-30"]    # 验证集：用于调参和早停
        self.test_time_range = ["2024-04-01", "2025-06-05"]   # 测试集：用于最终评估
        self.backtest_time_range = ["2024-07-01", "2025-06-05"] # 回测集：用于策略回测

        # TODO: Directory to save the processed, pickled datasets.
        # 预处理数据保存路径：处理好的 .pkl 文件会存放在这里，下次可以直接加载，无需重复处理
        self.dataset_path = "./data/processed_datasets"

        # =================================================================
        # Training Hyperparameters (训练超参数)
        # =================================================================
        self.clip = 5.0  # 截断值：防止极端异常值干扰训练

        self.epochs = 30 # 总训练轮数
        self.log_interval = 100  # 日志打印频率
        self.batch_size = 50  # 批次大小：显存够大可以调大

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        # 采样数：对于超大数据集，每个epoch只随机采样一部分数据进行训练
        self.n_train_iter = 2000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size

        # Learning rates for different model components.
        # 学习率：控制参数更新的步长
        self.tokenizer_learning_rate = 2e-4  # Tokenizer通常用较大的学习率
        self.predictor_learning_rate = 4e-5  # Predictor通常用较小的学习率微调

        # Gradient accumulation to simulate a larger batch size.
        # 梯度累积：显存不够时，累积几次梯度再更新，变相增大batch_size
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        # 优化器参数
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1 # 权重衰减：防止过拟合

        # Miscellaneous
        self.seed = 100  # 随机种子：保证实验可复现

        # =================================================================
        # Experiment Logging & Saving (实验记录与保存)
        # =================================================================
        self.use_comet = True # 是否使用 Comet ML 记录实验
        self.comet_config = {
            # It is highly recommended to load secrets from environment variables
            # for security purposes. Example: os.getenv("COMET_API_KEY")
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace" # TODO: Change to your Comet ML workspace name
        }
        self.comet_tag = 'finetune_demo'
        self.comet_name = 'finetune_demo'

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        # 模型保存路径
        self.save_path = "./outputs/models"
        self.tokenizer_save_folder_name = 'finetune_tokenizer_demo'
        self.predictor_save_folder_name = 'finetune_predictor_demo'
        self.backtest_save_folder_name = 'finetune_backtest_demo'

        # Path for backtesting results.
        # 回测结果保存路径
        self.backtest_result_path = "./outputs/backtest_results"

        # =================================================================
        # Model & Checkpoint Paths (模型路径)
        # =================================================================
        # TODO: Update these paths to your pretrained model locations.
        # These can be local paths or Hugging Face Hub model identifiers.
        # 预训练模型路径：从这里加载初始模型权重
        self.pretrained_tokenizer_path = "path/to/your/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "path/to/your/Kronos-small"

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = 5  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = 5  # Minimum holding period for a stock.
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")
