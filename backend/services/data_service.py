import yfinance as yf
import pandas as pd
from typing import Tuple, Optional

class DataService:
    @staticmethod
    def load_data(symbol: str, period: str, interval: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Load historical stock data from yfinance.
        Logic preserved from custom_predictor.py
        """
        try:
            df = yf.download(symbol, period=period, interval=interval)

            if df.empty:
                return df, f"No data found for symbol {symbol}"

            df = df.reset_index()

            # Handle MultiIndex columns safely
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Convert all column names to lowercase strings
            df.columns = [str(col).lower() for col in df.columns]

            # Normalize datetime column name
            if "date" in df.columns:
                df.rename(columns={"date": "datetime"}, inplace=True)
            elif "datetime" not in df.columns:
                df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

            # Ensure required columns exist
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    return pd.DataFrame(), f"Missing required column: {col}"

            # Kronos expects "amount"
            df["amount"] = df["close"] * df["volume"]
            
            # Ensure datetime is pd.Timestamp
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            return df, None
        except Exception as e:
            return pd.DataFrame(), str(e)
