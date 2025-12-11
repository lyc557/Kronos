import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
    """
    绘制预测结果对比图。
    
    如何通过这张图评价模型效果 (Evaluation Guide)：
    1. 趋势拟合 (Trend Following): 
       - 最重要的是看红色线(预测)是否跟住了蓝色线(真实)的大方向。
       - 只要涨跌趋势对上了，即使绝对数值有偏差也是可以接受的。
       
    2. 拐点捕捉 (Turning Points):
       - 观察在波峰和波谷处，红色线是否能及时转向。
       - 常见的不足是“滞后”(Lag)，即真实价格跌了，模型下一时刻才跟着跌。
       
    3. 波动幅度 (Volatility):
       - 红色线的波动幅度是否与蓝色线接近？
       - 如果红色线很平(趋于直线)，说明模型过于保守(欠拟合)；如果波动过大，可能是过拟合。
       
    4. 成交量 (Volume - 下方子图):
       - 关注模型是否预测出了成交量的放大和缩小，这往往对应着价格剧烈波动的时刻。
    """
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('prediction_result.png')
    print("预测结果图已保存至 prediction_result.png")


# 1. Load Model and Tokenizer
# print("1. 正在加载模型和分词器... (Tokenizer: NeoQuasar/Kronos-Tokenizer-base, Model: NeoQuasar/Kronos-base)")
# tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
# model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
print("1. 正在加载模型和分词器...")
tokenizer_path = "/home/ubuntu/chengtay_code/Kronos/finetune_csv/finetuned/HK_ali_09988_kline_5min_all/tokenizer/best_model"
model_path = "/home/ubuntu/chengtay_code/Kronos/finetune_csv/finetuned/HK_ali_09988_kline_5min_all/basemodel/best_model"

print(f"   Tokenizer Path: {tokenizer_path}")
print(f"   Model Path: {model_path}")

tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
model = Kronos.from_pretrained(model_path)

# 2. Instantiate Predictor
print("2. 正在初始化预测器... (Device: cuda:0, Max Context: 512)")
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
print("3. 正在准备数据... (Loading: ./data/XSHG_5min_600977.csv)")
df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])
print(f"   数据已加载，共 {len(df)} 行。")
print("   使用的特征列: ['open', 'high', 'low', 'close', 'volume', 'amount']")

lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 4. Make Prediction
print(f"4. 正在进行预测... (Lookback: {lookback}, Prediction Length: {pred_len})")
print(f"   参数设置: T=1.0, top_p=0.9, sample_count=1")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 5. Visualize Results
print("5. 正在可视化结果...")
print("预测数据前几行:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback+pred_len-1]

# visualize
plot_prediction(kline_df, pred_df)

