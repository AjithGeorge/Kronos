import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import traceback

# Ensure local Kronos import works
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_path)
print(f"📍 Added root to sys.path: {root_path}")

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("✅ Kronos models imported successfully")
except Exception as e:
    print(f"❌ Failed to import Kronos models: {e}")
    traceback.print_exc()
    Kronos = None
    KronosTokenizer = None
    KronosPredictor = None

KRONOS_MODELS = {
    "kronos-mini": {
        "name": "Kronos-mini",
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "context_length": 2048,
        "params": "4.1M",
        "description": "Lightweight model, suitable for fast prediction",
    },
    "kronos-small": {
        "name": "Kronos-small",
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "24.7M",
        "description": "Small model, balanced performance and speed",
    },
    "kronos-base": {
        "name": "Kronos-base",
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "102.3M",
        "description": "Base model, provides better prediction quality",
    },
}

class PredictionService:
    def __init__(self):
        self.loaded_models = {}

    def get_model(self, model_key: str):
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        if model_key not in KRONOS_MODELS:
            model_key = "kronos-base"

        config = KRONOS_MODELS[model_key]
        
        if Kronos is None:
            return None, config

        tokenizer = KronosTokenizer.from_pretrained(config["tokenizer_id"])
        model = Kronos.from_pretrained(config["model_id"])
        predictor = KronosPredictor(model, tokenizer, max_context=config["context_length"])

        self.loaded_models[model_key] = (predictor, config)
        return predictor, config

    def predict_single(self, model_key: str, df: pd.DataFrame, pred_len: int, lookback_limit: int, interval: str = "1d"):
        predictor, config = self.get_model(model_key)
        
        if predictor is None:
            raise ValueError(f"Model {model_key} could not be loaded.")

        total_rows = len(df)
        model_lookback = min(lookback_limit, config["context_length"], total_rows)

        x_df = df.tail(model_lookback)[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = df["datetime"].tail(model_lookback)

        # Generate future timestamps dynamically by inferring market hours from historical data
        last_date = df["datetime"].max()
        valid_times = sorted(list(set(df["datetime"].dt.time)))
        # Consider only 5 working days per week (Monday-Friday)
        valid_days = {0, 1, 2, 3, 4}

        future_ts = []
        curr_date = last_date.normalize()
        max_days = pred_len * 10
        days_checked = 0

        while len(future_ts) < pred_len and days_checked < max_days:
            if curr_date.dayofweek in valid_days:
                for t in valid_times:
                    try:
                        ts = pd.Timestamp(
                            year=curr_date.year, 
                            month=curr_date.month, 
                            day=curr_date.day, 
                            hour=t.hour, 
                            minute=t.minute,
                            second=t.second,
                            tz=last_date.tz
                        )
                        if ts > last_date:
                            future_ts.append(ts)
                            if len(future_ts) == pred_len:
                                break
                    except Exception:
                        pass
            
            if len(future_ts) == pred_len:
                break
                
            curr_date += pd.Timedelta(days=1)
            days_checked += 1

        # Fallback if inference failed to generate enough points
        if len(future_ts) < pred_len:
            remaining = pred_len - len(future_ts)
            fallback_ts = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=remaining, 
                freq="B"
            )
            future_ts.extend(list(fallback_ts))

        y_timestamp = pd.Series(future_ts)

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
        
        # Ensure index is datetime for consistency
        pred_df.index = y_timestamp
        pred_df.index.name = "datetime"

        return {
            "model_key": model_key,
            "pred_df": pred_df,
            "config": config,
            "lookback_used": model_lookback,
            "y_timestamp": y_timestamp
        }

    def predict_parallel(self, model_keys: List[str], df: pd.DataFrame, pred_len: int, lookback_limit: int, interval: str = "1d"):
        """Run multiple models in parallel using threads."""
        results = {}
        
        # Using ThreadPoolExecutor because Kronos/Torch usually releases GIL during compute
        # and we avoid the overhead of moving models across processes.
        with ThreadPoolExecutor(max_workers=len(model_keys)) as executor:
            future_to_model = {
                executor.submit(self.predict_single, m, df, pred_len, lookback_limit, interval): m 
                for m in model_keys
            }
            
            for future in future_to_model:
                model_key = future_to_model[future]
                try:
                    res = future.result()
                    results[model_key] = res
                except Exception as e:
                    print(f"Error predicting with {model_key}: {e}")
                    results[model_key] = {"error": str(e)}
        
        return results

    def run_backtest(self, model_key: str, df: pd.DataFrame, lookback: int = 400, pred_len: int = 40):
        predictor, config = self.get_model(model_key)
        
        if predictor is None:
            raise ValueError(f"Model {model_key} could not be loaded.")

        df = df.copy()
        df = df.sort_values("datetime").reset_index(drop=True)

        # Split data - use last pred_len points as test set
        train_df = df.iloc[:-pred_len]
        test_df = df.iloc[-pred_len:]

        # Context window from training data
        model_lookback = min(lookback, config["context_length"], len(train_df))
        x_df = train_df.tail(model_lookback)[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = train_df["datetime"].tail(model_lookback)

        # Ground truth timestamps
        y_timestamp = test_df["datetime"].reset_index(drop=True)

        # Prediction
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

        # Align indices
        pred_df.index = y_timestamp
        pred_df.index.name = "datetime"
        test_df = test_df.set_index("datetime")

        # Calculate metrics
        metrics = self.calculate_backtest_metrics(pred_df, test_df)

        return {
            "model_key": model_key,
            "pred_df": pred_df,
            "actual_df": test_df,
            "metrics": metrics,
            "config": config,
            "pred_len": pred_len
        }

    def calculate_backtest_metrics(self, pred_df, actual_df):
        results = {}
        for col in ["close", "volume"]:
            if col in pred_df.columns and col in actual_df.columns:
                y_true = actual_df[col]
                y_pred = pred_df[col]

                # Align indexes
                y_true, y_pred = y_true.align(y_pred, join="inner")

                if len(y_true) > 0:
                    mae = np.mean(np.abs(y_true - y_pred))
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100

                    results[col] = {
                        "MAE": float(mae),
                        "RMSE": float(rmse),
                        "MAPE (%)": float(mape),
                        "Count": int(len(y_true)),
                    }
        return results

    def run_backtest_all(self, model_keys: List[str], df: pd.DataFrame, lookback: int = 256, pred_len: int = 30):
        """Run backtests for all selected models and return consolidated results."""
        results = {}
        for model_key in model_keys:
            try:
                res = self.run_backtest(model_key, df, lookback, pred_len)
                results[model_key] = res
            except Exception as e:
                print(f"Backtest failed for {model_key}: {e}")
                results[model_key] = {"error": str(e)}
        return results
