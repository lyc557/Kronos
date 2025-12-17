import os
import pickle
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from tqdm import trange

from config import Config


class QlibDataPreprocessor:
    """
    A class to handle the loading, processing, and splitting of Qlib financial data.
    """

    def __init__(self):
        """Initializes the preprocessor with configuration and data fields."""
        self.config = Config()
        # 这些字段是从 Qlib 原始行情中提取的基础价格与成交信息
        # 作为量化任务的输入特征，后续会被加工成模型可用的矩阵
        self.data_fields = ['open', 'close', 'high', 'low', 'volume', 'vwap']
        self.data = {}  # A dictionary to store processed data for each symbol.

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        print("Initializing Qlib...")
        # 初始化 Qlib 数据提供器
        # provider_uri 指向已经准备好的日频/分钟频数据目录
        # region=REG_CN 指定中国市场（沪深）的数据与交易日历
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def load_qlib_data(self):
        """
        从 Qlib 加载原始数据，逐个股票进行处理，并存入 `self.data`。
        处理流程：
        1. 获取交易日历，确定起止时间。
        2. 使用 QlibDataLoader 加载原始数据（Open, High, Low, Close, Volume, VWAP）。
        3. 对每个股票：
           - 格式转换：重命名列，计算成交额 (Amount)。
           - 特征筛选：只保留 config 中定义的特征。
           - 长度过滤：剔除数据长度不足（小于 lookback + predict）的股票。
        """
        print("Loading and processing data from Qlib...")
        data_fields_qlib = ['$' + f for f in self.data_fields]
        # Qlib 的交易日历（或时间戳序列），用于精确切片数据时间范围
        cal: np.ndarray = D.calendar()

        # Determine the actual start and end times to load, including buffer for lookback and predict windows.
        # 确定实际加载的时间范围：需要额外向前多取 lookback_window，向后多取 predict_window
        start_index = cal.searchsorted(pd.Timestamp(self.config.dataset_begin_time))
        end_index = cal.searchsorted(pd.Timestamp(self.config.dataset_end_time))

        # Check if start_index lookbackw_window will cause negative index
        # 为了构造滑动窗口，训练输入需要“向前看”一段上下文长度
        # 因此从 begin_time 对应索引再向前偏移 lookback_window 作为真实加载起点
        adjusted_start_index = max(start_index - self.config.lookback_window, 0)
        real_start_time = cal[adjusted_start_index]

        # Check if end_index exceeds the range of the array
        if end_index >= len(cal):
            end_index = len(cal) - 1
        elif cal[end_index] != pd.Timestamp(self.config.dataset_end_time):
            end_index -= 1

        # Check if end_index+predictw_window will exceed the range of the array
        # 同理，真实加载终点需要向后偏移 predict_window，保证预测期数据可用
        adjusted_end_index = min(end_index + self.config.predict_window, len(cal) - 1)
        real_end_time = cal[adjusted_end_index]

        # Load data using Qlib's data loader.
        # QlibDataLoader 返回一个 MultiIndex 的 DataFrame：
        # 行索引包含 ['datetime', 'instrument']，列为字段（如 $open/$close）
        # 这里按真实起止时间范围进行一次性加载
        data_df = QlibDataLoader(config=data_fields_qlib).load(
            self.config.instrument, real_start_time, real_end_time
        )

        # --- Debug Info ---
        if not data_df.empty:
            print(f"\n[Debug] Raw Qlib Data Loaded. Rows: {len(data_df)}")
            try:
                if 'datetime' in data_df.index.names:
                    dates = data_df.index.get_level_values('datetime')
                    print(f"[Debug] Time Range: {dates.min()} to {dates.max()}")
                if 'instrument' in data_df.index.names:
                    insts = data_df.index.get_level_values('instrument').unique()
                    print(f"[Debug] Instruments Loaded: {len(insts)}")
                    print(f"[Debug] Sample Instruments: {insts[:5].tolist()}")
            except Exception as e:
                print(f"[Debug] Error inspecting index: {e}")
        else:
            print("\n[Debug] Warning: QlibDataLoader returned empty data!")
        # ------------------

        # 将列层级（字段）压栈为行，再按第二层级展开
        # 目标形态：列为“股票代码”，行索引为时间，单元格保存某字段取值
        # 后续会逐股票进行透视，拼成统一的特征矩阵
        data_df = data_df.stack().unstack(level=1)  # Reshape for easier access.

        symbol_list = list(data_df.columns)
        for i in trange(len(symbol_list), desc="Processing Symbols"):
            symbol = symbol_list[i]
            symbol_df = data_df[symbol]

            # Pivot the table to have features as columns and datetime as index.
            # 将“字段”变成列，“datetime”作为索引，得到每个时间点的多维特征
            symbol_df = symbol_df.reset_index().rename(columns={'level_1': 'field'})
            symbol_df = pd.pivot(symbol_df, index='datetime', columns='field', values=symbol)
            symbol_df = symbol_df.rename(columns={f'${field}': field for field in self.data_fields})

            # Calculate amount and select final features.
            # 构造两个常用的交易特征：
            # - vol：成交量（与原始 volume 同义）
            # - amt：近似的成交额（用 OHLC 的均值乘以成交量），衡量资金流大小
            symbol_df['vol'] = symbol_df['volume']
            symbol_df['amt'] = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4 * symbol_df['vol']
            symbol_df = symbol_df[self.config.feature_list]

            # Filter out symbols with insufficient data.
            # 丢弃存在缺失值的时间点，并过滤掉长度不足以支撑“上下文 + 预测期”窗口的股票
            symbol_df = symbol_df.dropna()
            if len(symbol_df) < self.config.lookback_window + self.config.predict_window + 1:
                continue

            self.data[symbol] = symbol_df

        print(f"\n[Debug] Processing Complete. Total valid symbols: {len(self.data)}")
        if self.data:
            sample_sym = list(self.data.keys())[0]
            print(f"[Debug] Example Symbol '{sample_sym}' Date Range: {self.data[sample_sym].index.min()} to {self.data[sample_sym].index.max()}")

    def prepare_dataset(self):
        """
        Splits the loaded data into train, validation, and test sets and saves them to disk.
        """
        print("Splitting data into train, validation, and test sets...")
        train_data, val_data, test_data = {}, {}, {}

        symbol_list = list(self.data.keys())
        for i in trange(len(symbol_list), desc="Preparing Datasets"):
            symbol = symbol_list[i]
            symbol_df = self.data[symbol]

            # Define time ranges from config.
            # 使用配置中定义的时间区间进行时序切分：
            # - 训练集：用于参数学习
            # - 验证集：用于调参与早停
            # - 测试集：仅用于最终评估
            train_start, train_end = self.config.train_time_range
            val_start, val_end = self.config.val_time_range
            test_start, test_end = self.config.test_time_range

            # Create boolean masks for each dataset split.
            # 通过布尔掩码进行切片，保持各股票在相同时间区间的对齐
            train_mask = (symbol_df.index >= train_start) & (symbol_df.index <= train_end)
            val_mask = (symbol_df.index >= val_start) & (symbol_df.index <= val_end)
            test_mask = (symbol_df.index >= test_start) & (symbol_df.index <= test_end)

            # Apply masks to create the final datasets.
            train_data[symbol] = symbol_df[train_mask]
            val_data[symbol] = symbol_df[val_mask]
            test_data[symbol] = symbol_df[test_mask]

        # Save the datasets using pickle.
        # 将三份数据以 pickle 序列化保存，后续训练/回测脚本直接加载使用
        os.makedirs(self.config.dataset_path, exist_ok=True)
        with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        print("Datasets prepared and saved successfully.")


if __name__ == '__main__':
    # This block allows the script to be run directly to perform data preprocessing.
    preprocessor = QlibDataPreprocessor()
    preprocessor.initialize_qlib()
    preprocessor.load_qlib_data()
    preprocessor.prepare_dataset()

