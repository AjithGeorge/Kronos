import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Ensure the root directory is in sys.path to find the model module
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
    # Ensure both dataframes use timestamps as their index for proper alignment on the X-axis
    kline_df = kline_df.set_index("timestamps")
    # pred_df index is already timestamps from the predictor

    sr_close = kline_df["close"]
    sr_pred_close = pred_df["close"]
    sr_close.name = "History"
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df["volume"]
    sr_pred_volume = pred_df["volume"]
    sr_volume.name = "History"
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df["History"], label="History", color="blue", linewidth=1.5)
    ax1.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df["History"], label="History", color="blue", linewidth=1.5)
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("prediction_result.png")
    print("\nPrediction plot saved to prediction_result.png")
    # plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 3. Prepare Data
input_file = "./data/RECLTD_kronos.csv"
print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

# Parse timestamps and sort chronologically
df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)

# Adjust lookback and pred_len based on available data
total_rows = len(df)
lookback = min(512, total_rows)  # Use up to 512 for context
pred_len = 30  # Forecast the next 30 business days

print(f"Total rows: {total_rows}, using lookback: {lookback}, pred_len: {pred_len}")

# Use the latest available data as context for the future prediction
x_df = df.tail(lookback)[["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df["timestamps"].tail(lookback)

# Generate actual future timestamps (Business days) starting from the day after the last record
last_date = df["timestamps"].max()
y_timestamp = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=pred_len, freq="B"
)
y_timestamp = pd.Series(y_timestamp)

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical context and forecasted data for plotting
# We show the last 100 days of history for context in the plot
kline_df = df.tail(100)

# visualize
plot_prediction(kline_df, pred_df)
