import yfinance as yf
import pandas as pd


def fetch_and_save_kronos_csv(symbol, interval="1d", period="5y", filename=None):
    ticker = f"{symbol}.NS"

    df = yf.download(ticker, interval=interval, period=period)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    df = df.reset_index()

    # Normalize column names
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    # Convert timestamp → desired string format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Format: YYYY/MM/DD H:MM (no leading zero in hour)
    df["timestamp"] = df["timestamp"].apply(
        lambda x: f"{x.year}/{x.month:02d}/{x.day:02d} {x.hour}:{x.minute:02d}"
    )

    # Add amount column
    df["amount"] = df["close"] * df["volume"]

    # Keep required columns
    df = df[["timestamp", "open", "high", "low", "close", "volume", "amount"]]

    df = df.dropna()
    df = df.sort_index()

    if filename is None:
        filename = f"{symbol}_kronos.csv"

    df.to_csv(filename, index=False)

    return df


# Example
df = fetch_and_save_kronos_csv("RECLTD")
print(df.head())
