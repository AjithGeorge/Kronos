import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from model import Kronos, KronosTokenizer, KronosPredictor


# =========================
# 📊 Metrics
# =========================
def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


# =========================
# 🔁 Walk-forward backtest
# =========================
def walk_forward_backtest(df, predictor, lookback=512, pred_len=30, step=10):
    """
    step = how much we shift forward each iteration
    smaller step = more accurate evaluation, slower runtime
    """

    df = df.copy().sort_values("timestamps").reset_index(drop=True)

    results = []
    all_preds = []
    all_actuals = []

    start = lookback
    end = len(df) - pred_len

    print(f"\nRunning walk-forward backtest...")
    print(f"Total windows: {(end - start) // step}")

    for i in range(start, end, step):

        train_df = df.iloc[:i]
        test_df = df.iloc[i : i + pred_len]

        if len(test_df) < pred_len:
            break

        x_df = train_df.tail(lookback)[
            ["open", "high", "low", "close", "volume", "amount"]
        ]
        x_timestamp = train_df["timestamps"].tail(lookback)

        y_timestamp = test_df["timestamps"].reset_index(drop=True)

        # Predict
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

        pred_df.index = y_timestamp
        actual_df = test_df.set_index("timestamps")

        # Store
        all_preds.append(pred_df)
        all_actuals.append(actual_df)

        # Metrics per window (close only is usually enough)
        mae, rmse, mape = compute_metrics(
            actual_df["close"].values, pred_df["close"].values
        )

        results.append(
            {
                "window_start": train_df["timestamps"].iloc[-1],
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
            }
        )

        print(f"Window {i}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

    return pd.DataFrame(results), pd.concat(all_preds), pd.concat(all_actuals)


# =========================
# 📈 Plot full comparison
# =========================
def plot_walk_forward(actual, pred):
    plt.figure(figsize=(12, 5))

    plt.plot(actual.index, actual["close"], label="Actual", color="blue")
    plt.plot(pred.index, pred["close"], label="Predicted", color="red", alpha=0.7)

    plt.title("Walk-Forward Backtest - Close Price")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("walk_forward_result.png")
    print("\nSaved: walk_forward_result.png")


# =========================
# 🚀 MAIN
# =========================

print("Loading model...")

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

predictor = KronosPredictor(model, tokenizer, max_context=512)


# Load data
input_file = "./data/RECLTD_kronos.csv"
df = pd.read_csv(input_file)

df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)

print(f"Total data points: {len(df)}")


# Config
LOOKBACK = min(512, len(df))
PRED_LEN = 30
STEP = 10  # smaller = stricter backtest


# Run walk-forward
metrics_df, pred_all, actual_all = walk_forward_backtest(
    df, predictor, lookback=LOOKBACK, pred_len=PRED_LEN, step=STEP
)


# =========================
# 📊 Summary stats
# =========================
print("\n===== OVERALL RESULTS =====")
print(metrics_df.describe())


# Average performance
print("\nAverage Metrics:")
print(metrics_df[["MAE", "RMSE", "MAPE"]].mean())


# =========================
# 📈 Plot
# =========================
actual_all = actual_all.sort_index()
pred_all = pred_all.sort_index()

plot_walk_forward(actual_all, pred_all)
