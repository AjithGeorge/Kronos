import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure local module import
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from model import Kronos, KronosTokenizer, KronosPredictor


# =========================
# 📊 Plotting
# =========================
def plot_backtest(actual_df, pred_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # CLOSE
    ax1.plot(actual_df.index, actual_df["close"], label="Actual", color="blue")
    ax1.plot(pred_df.index, pred_df["close"], label="Predicted", color="red")
    ax1.set_title("Backtest - Close Price")
    ax1.legend()
    ax1.grid(True)

    # VOLUME
    ax2.plot(actual_df.index, actual_df["volume"], label="Actual", color="blue")
    ax2.plot(pred_df.index, pred_df["volume"], label="Predicted", color="red")
    ax2.set_title("Backtest - Volume")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("backtest_result.png")
    print("\nBacktest plot saved to backtest_result.png")


# =========================
# 📈 Metrics
# =========================
def evaluate_predictions(pred_df, actual_df):
    results = {}

    for col in ["close", "volume"]:
        y_true = actual_df[col]
        y_pred = pred_df[col]

        # Align indexes
        y_true, y_pred = y_true.align(y_pred, join="inner")

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        results[col] = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

    return results


# =========================
# 🔁 Backtest Logic
# =========================
def run_backtest(df, predictor, lookback=512, pred_len=30):
    print(f"\nRunning backtest for last {pred_len} steps...")

    df = df.copy()
    df = df.sort_values("timestamps").reset_index(drop=True)

    # Split
    train_df = df.iloc[:-pred_len]
    test_df = df.iloc[-pred_len:]

    # Context window
    x_df = train_df.tail(lookback)[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = train_df["timestamps"].tail(lookback)

    # Ground truth timestamps
    y_timestamp = test_df["timestamps"].reset_index(drop=True)

    # Prediction
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

    # Align indices
    pred_df.index = y_timestamp
    test_df = test_df.set_index("timestamps")

    return pred_df, test_df


# =========================
# 🚀 MAIN
# =========================

# 1. Load model
print("Loading Kronos model...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

predictor = KronosPredictor(model, tokenizer, max_context=512)


# 2. Load data
input_file = "./data/RECLTD_kronos.csv"
print(f"Loading data from {input_file}...")

df = pd.read_csv(input_file)

# Parse timestamps
df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)

print(f"Total rows: {len(df)}")


# 3. Config
LOOKBACK = min(512, len(df))
PRED_LEN = 30  # change to 45 if needed


# 4. Run backtest
pred_df, actual_df = run_backtest(df, predictor, lookback=LOOKBACK, pred_len=PRED_LEN)


# 5. Print samples
print("\nActual Data (head):")
print(actual_df.head())

print("\nPredicted Data (head):")
print(pred_df.head())


# 6. Metrics
metrics = evaluate_predictions(pred_df, actual_df)

print("\n📊 Backtest Metrics:")
for feature, vals in metrics.items():
    print(f"\n{feature.upper()}:")
    for k, v in vals.items():
        print(f"  {k}: {v:.4f}")


# 7. Plot
plot_backtest(actual_df, pred_df)
