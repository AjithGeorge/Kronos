import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import sys
import os

# Ensure local Kronos import works
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from model import Kronos, KronosTokenizer, KronosPredictor


# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Kronos Stock Predictor", layout="wide")

st.title("📈 Kronos Time-Series Forecasting App")

# Sidebar inputs
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
period = st.sidebar.selectbox(
    "Historical Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3,
)
interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1h", "30m"],
    index=0,
)

pred_len = st.sidebar.slider("Prediction Length (days)", 5, 60, 30)
lookback_limit = st.sidebar.slider("Lookback Window", 100, 512, 256)


# -----------------------------
# LOAD DATA FROM YFINANCE
# -----------------------------
@st.cache_data
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)

    if df.empty:
        return df

    df = df.reset_index()

    # ✅ Handle MultiIndex columns safely
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # ✅ Convert all column names to lowercase strings
    df.columns = [str(col).lower() for col in df.columns]

    # ✅ Normalize datetime column name
    if "date" in df.columns:
        df.rename(columns={"date": "datetime"}, inplace=True)
    elif "datetime" not in df.columns:
        df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

    # ✅ Ensure required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Kronos expects "amount"
    df["amount"] = df["close"] * df["volume"]

    return df


# -----------------------------
# DATA LOADING (DEFERRED)
# -----------------------------

# Initialize session state for data
if "df" not in st.session_state:
    st.session_state.df = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Button to load data
if st.sidebar.button("Load Data", type="primary"):
    with st.spinner(f"Loading data for {symbol}..."):
        df = load_data(symbol, period, interval)

        if df.empty:
            st.error("No data found. Try another symbol.")
        else:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("Data loaded successfully!")

# Check if data is loaded
if not st.session_state.data_loaded:
    st.info("👈 Please enter a stock symbol and click 'Load Data' in the sidebar.")
    st.stop()

df = st.session_state.df

st.subheader("📊 Raw Data")
st.dataframe(df.tail())


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    return predictor


predictor = load_model()


# -----------------------------
# PREPARE DATA
# -----------------------------
total_rows = len(df)
lookback = min(lookback_limit, total_rows)

x_df = df.tail(lookback)[["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df["datetime"].tail(lookback)

last_date = df["datetime"].max()

y_timestamp = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=pred_len,
    freq="B",
)

y_timestamp = pd.Series(y_timestamp)


# -----------------------------
# PREDICT
# -----------------------------
if st.button("🔮 Run Prediction"):

    with st.spinner("Running Kronos prediction..."):
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

    st.success("Prediction complete!")

    # -----------------------------
    # PLOTTING (INTERACTIVE)
    # -----------------------------
    st.subheader("📈 Interactive Forecast")

    hist_df = df.tail(150).copy()
    hist_df = hist_df.set_index("datetime")

    pred_df.index = pd.to_datetime(y_timestamp)

    # ---- PRICE CHART ----
    fig_price = go.Figure()

    # Historical Candlestick
    fig_price.add_trace(
        go.Candlestick(
            x=hist_df.index,
            open=hist_df["open"],
            high=hist_df["high"],
            low=hist_df["low"],
            close=hist_df["close"],
            name="Historical",
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    )

    # ✅ Ensure prediction has all OHLC columns
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in pred_df.columns:
            pred_df[col] = pred_df["close"]  # fallback

    # Prediction Candlestick
    fig_price.add_trace(
        go.Candlestick(
            x=pred_df.index,
            open=pred_df["open"],
            high=pred_df["high"],
            low=pred_df["low"],
            close=pred_df["close"],
            name="Prediction",
            increasing_line_color="blue",
            decreasing_line_color="orange",
            opacity=0.7,
        )
    )

    # Optional: separator line between history & prediction
    fig_price.add_vline(
        x=hist_df.index[-1], line_width=2, line_dash="dash", line_color="white"
    )

    fig_price.update_layout(
        title=f"{symbol} Price Forecast (OHLC)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=True,
    )

    st.plotly_chart(fig_price, use_container_width=True)

    # ---- VOLUME CHART ----
    fig_vol = go.Figure()

    fig_vol.add_trace(
        go.Bar(
            x=hist_df.index,
            y=hist_df["volume"],
            name="Historical Volume",
            marker_color="gray",
        )
    )

    fig_vol.add_trace(
        go.Bar(
            x=pred_df.index,
            y=pred_df["volume"],
            name="Predicted Volume",
            marker_color="orange",
        )
    )

    fig_vol.update_layout(
        title="Volume Forecast",
        template="plotly_dark",
        height=400,
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    # ---- METRICS ----
    st.subheader("📊 Forecast Summary")

    last_price = hist_df["close"].iloc[-1]
    future_price = pred_df["close"].iloc[-1]
    min_pred = pred_df["low"].min()
    max_pred = pred_df["high"].max()
    avg_pred_vol = pred_df["volume"].mean()
    last_vol = hist_df["volume"].iloc[-1]

    change_pct = ((future_price - last_price) / last_price) * 100
    pred_range = max_pred - min_pred
    vol_change_pct = ((avg_pred_vol - last_vol) / last_vol) * 100

    # Determine trend direction and color indicators
    trend = (
        "📈 Bullish"
        if change_pct > 0
        else "📉 Bearish" if change_pct < 0 else "➡️ Neutral"
    )
    trend_delta = f"{change_pct:+.2f}%" if change_pct != 0 else "0.00%"

    price_delta = f"{change_pct:+.2f}%" if change_pct != 0 else "0.00%"
    high_delta = f"{((max_pred - last_price) / last_price) * 100:+.2f}%"
    low_delta = f"{((min_pred - last_price) / last_price) * 100:+.2f}%"

    # Display KPI cards in rows
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Current Price",
        f"{last_price:.2f}",
        help="Last closing price from historical data",
    )
    col2.metric(
        "Predicted Price",
        f"{future_price:.2f}",
        price_delta,
        help="Predicted closing price at end of forecast period",
    )
    col3.metric(
        "Predicted High",
        f"{max_pred:.2f}",
        high_delta,
        help="Highest predicted price in forecast period",
    )
    col4.metric(
        "Predicted Low",
        f"{min_pred:.2f}",
        low_delta,
        help="Lowest predicted price in forecast period",
    )

    # Second row of metrics
    col5, col6, col7, col8 = st.columns(4)
    col5.metric(
        "Price Range",
        f"{pred_range:.2f}",
        help="Spread between highest and lowest predicted prices",
    )
    col6.metric(
        "Trend",
        trend,
        trend_delta,
        help="Overall price direction based on prediction",
    )
    col7.metric(
        "Avg Predicted Volume",
        f"{avg_pred_vol:,.0f}",
        f"{vol_change_pct:+.2f}%",
        help="Average daily volume in forecast vs last historical",
    )
    col8.metric(
        "Forecast Period",
        f"{len(pred_df)} days",
        help="Number of days in prediction horizon",
    )

    # Add a visual separator
    st.markdown("---")
