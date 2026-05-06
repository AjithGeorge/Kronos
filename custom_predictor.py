import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import sys
import os

# Ensure local Kronos import works
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from model import Kronos, KronosTokenizer, KronosPredictor

# =========================
# MODEL CONFIGURATIONS
# =========================
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


# =========================
# BACKTEST FUNCTIONS
# =========================


def calculate_backtest_metrics(pred_df, actual_df):
    """Calculate MAE, RMSE, MAPE for backtest predictions"""
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
                # Avoid division by zero in MAPE
                mape = (
                    np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
                )

                results[col] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE (%)": mape,
                    "Count": len(y_true),
                }

    return results


def run_backtest(df, predictor, lookback=512, pred_len=30):
    """Run backtest by predicting the last pred_len points"""
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Split data - use last pred_len points as test set
    train_df = df.iloc[:-pred_len]
    test_df = df.iloc[-pred_len:]

    # Context window from training data
    x_df = train_df.tail(lookback)[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = train_df["datetime"].tail(lookback)

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
    test_df = test_df.set_index("datetime")

    return pred_df, test_df


def plot_backtest_results(actual_df, pred_df):
    """Create interactive Plotly charts for backtest results"""
    fig = go.Figure()

    # Actual close price
    fig.add_trace(
        go.Scatter(
            x=actual_df.index,
            y=actual_df["close"],
            mode="lines+markers",
            name="Actual Close",
            line=dict(color="blue", width=2),
        )
    )

    # Predicted close price
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df["close"],
            mode="lines+markers",
            name="Predicted Close",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Backtest: Actual vs Predicted Close Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
        legend=dict(x=0, y=1),
    )

    return fig


# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Kronos Stock Predictor", layout="wide")

st.title("📈 Kronos Time-Series Forecasting App")

# Sidebar inputs
symbol = st.sidebar.text_input("Stock Symbol", value="RECLTD.NS")
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
if "prediction_run" not in st.session_state:
    st.session_state.prediction_run = False
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None
if "all_predictions" not in st.session_state:
    st.session_state.all_predictions = {}
if "hist_df" not in st.session_state:
    st.session_state.hist_df = None
if "y_timestamp" not in st.session_state:
    st.session_state.y_timestamp = None
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "backtest_run_all" not in st.session_state:
    st.session_state.backtest_run_all = False
if "backtest_results_all" not in st.session_state:
    st.session_state.backtest_results_all = {}

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
def load_model(model_key="kronos-base"):
    """Load a specific Kronos model and tokenizer"""
    if model_key not in KRONOS_MODELS:
        model_key = "kronos-base"

    config = KRONOS_MODELS[model_key]

    tokenizer = KronosTokenizer.from_pretrained(config["tokenizer_id"])
    model = Kronos.from_pretrained(config["model_id"])
    predictor = KronosPredictor(model, tokenizer, max_context=config["context_length"])

    return predictor, config


# Initialize session state for models
if "loaded_models" not in st.session_state:
    st.session_state.loaded_models = {}
if "selected_models" not in st.session_state:
    st.session_state.selected_models = ["kronos-base"]  # Default selection

# Model selection using Streamlit pills (native chip component)
st.sidebar.markdown("### 🎯 Select Models to Compare")

# Create options list with model names and keys
model_options = list(KRONOS_MODELS.keys())
model_labels = [
    f"{KRONOS_MODELS[k]['name']} ({KRONOS_MODELS[k]['params']})" for k in model_options
]

# Use st.pills for native chip selection (Streamlit 1.56+)
# Set default to kronos-base label
default_label = next(
    (label for label in model_labels if "kronos-base" in label.lower()), model_labels[0]
)

selected_labels = st.sidebar.pills(
    "Select models:",
    options=model_labels,
    default=[default_label],  # Always default to kronos-base
    selection_mode="multi",
    key="model_pill_selector",
)

# Convert selection back to model keys
if selected_labels:
    selected_models = []
    for label in selected_labels:
        # Extract model key from label (format: "Name (params)")
        for k in model_options:
            if f"{KRONOS_MODELS[k]['name']} ({KRONOS_MODELS[k]['params']})" == label:
                selected_models.append(k)
                break
    st.session_state.selected_models = selected_models
else:
    st.session_state.selected_models = ["kronos-base"]
    selected_models = ["kronos-base"]

selected_models = st.session_state.selected_models

# Display selected models summary in sidebar
if selected_models:
    st.sidebar.markdown("### 📊 Selected Models:")
    for model_key in selected_models:
        config = KRONOS_MODELS[model_key]
        st.sidebar.caption(
            f"✅ **{config['name']}** - Context: {config['context_length']} | {config['params']}"
        )

# Load models on demand
for model_key in selected_models:
    if model_key not in st.session_state.loaded_models:
        with st.spinner(f"Loading {KRONOS_MODELS[model_key]['name']}..."):
            predictor, config = load_model(model_key)
            st.session_state.loaded_models[model_key] = (predictor, config)

available_predictors = st.session_state.loaded_models


# -----------------------------
# PREPARE DATA
# -----------------------------
total_rows = len(df)

# Dynamically adjust lookback_limit based on available models
max_context = max([KRONOS_MODELS[k]["context_length"] for k in selected_models])
effective_lookback_limit = min(lookback_limit, max_context)

lookback = min(effective_lookback_limit, total_rows)

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
if st.button("🔮 Run Predictions with All Selected Models"):
    with st.spinner("Running Kronos predictions..."):
        all_predictions = {}

        for model_key in selected_models:
            predictor, config = available_predictors[model_key]

            # Adjust lookback based on model's context length
            model_lookback = min(lookback, config["context_length"], total_rows)

            # Prepare data for this model
            x_df = df.tail(model_lookback)[
                ["open", "high", "low", "close", "volume", "amount"]
            ]
            x_timestamp = df["datetime"].tail(model_lookback)

            try:
                # Run prediction
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

                all_predictions[model_key] = {
                    "pred_df": pred_df,
                    "config": config,
                    "lookback_used": model_lookback,
                }
            except Exception as e:
                st.error(f"Error running prediction with {config['name']}: {str(e)}")
                continue

        # Save to session state
        if all_predictions:
            st.session_state.all_predictions = all_predictions
            st.session_state.y_timestamp = y_timestamp
            st.session_state.hist_df = df.tail(150).copy().set_index("datetime")
            st.session_state.prediction_run = True
            st.success(f"✅ Predictions complete for {len(all_predictions)} model(s)!")
        else:
            st.error("No successful predictions. Please check the errors above.")

# Display prediction results if prediction has been run
if st.session_state.prediction_run and "all_predictions" in st.session_state:
    all_predictions = st.session_state.all_predictions
    y_timestamp = st.session_state.y_timestamp
    hist_df = st.session_state.hist_df

    if not all_predictions:
        st.warning("No predictions available. Please run predictions first.")
        st.stop()

    # Create tabs for each model
    tab_list = [KRONOS_MODELS[key]["name"] for key in all_predictions.keys()]
    tabs = st.tabs(tab_list)

    # Display results for each model in separate tabs
    for tab_idx, (model_key, predictions) in enumerate(all_predictions.items()):
        with tabs[tab_idx]:
            pred_df = predictions["pred_df"]
            config = predictions["config"]
            lookback_used = predictions["lookback_used"]

            # Ensure proper indexing
            pred_df.index = pd.to_datetime(y_timestamp)

            # Display model information
            st.info(
                f"**{config['name']}**\n\n"
                f"- Context Length: {config['context_length']} | Params: {config['params']}\n"
                f"- Lookback Used: {lookback_used} points\n"
                f"- {config['description']}"
            )

            # ---- PRICE CHART ----
            st.subheader("📈 Interactive Forecast")
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

            # Ensure prediction has all OHLC columns
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
                title=f"{symbol} Price Forecast (OHLC) - {config['name']}",
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

            # ---- DETAILED DATA TABLE ----
            st.subheader("📋 Prediction Details")
            with st.expander("View detailed prediction data"):
                st.dataframe(
                    pred_df[["open", "high", "low", "close", "volume"]].style.format(
                        {
                            "open": "{:.4f}",
                            "high": "{:.4f}",
                            "low": "{:.4f}",
                            "close": "{:.4f}",
                            "volume": "{:.0f}",
                        }
                    )
                )

    # ---- COMPARISON ACROSS MODELS (Optional) ----
    if len(all_predictions) > 1:
        st.markdown("---")
        st.subheader("🔄 Model Comparison")

        # Create comparison dataframe
        comparison_data = []
        for model_key, predictions in all_predictions.items():
            pred_df = predictions["pred_df"]
            config = predictions["config"]

            comparison_data.append(
                {
                    "Model": config["name"],
                    "Context Length": config["context_length"],
                    "Parameters": config["params"],
                    "Final Price": f"{pred_df['close'].iloc[-1]:.4f}",
                    "Min Predicted": f"{pred_df['low'].min():.4f}",
                    "Max Predicted": f"{pred_df['high'].max():.4f}",
                    "Avg Volume": f"{pred_df['volume'].mean():,.0f}",
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Comparison chart
        st.subheader("📊 Price Predictions Comparison")
        fig_comparison = go.Figure()

        # Add forward predictions for each model
        for model_key, predictions in all_predictions.items():
            pred_df = predictions["pred_df"]
            config = predictions["config"]

            fig_comparison.add_trace(
                go.Scatter(
                    x=pred_df.index,
                    y=pred_df["close"],
                    mode="lines+markers",
                    name=f"{config['name']} (Forward)",
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

        # Add historical data for reference
        fig_comparison.add_trace(
            go.Scatter(
                x=hist_df.index,
                y=hist_df["close"],
                mode="lines",
                name="Historical",
                line=dict(color="gray", dash="dash", width=2),
            )
        )

        # Add backtest predictions and actual values if available
        if st.session_state.get("backtest_run_all", False) and st.session_state.get(
            "backtest_results_all"
        ):
            backtest_results_all = st.session_state.backtest_results_all

            # Add backtest predictions for each model
            for model_key, backtest_result in backtest_results_all.items():
                config = backtest_result["config"]
                backtest_pred_df = backtest_result["pred_df"]

                fig_comparison.add_trace(
                    go.Scatter(
                        x=backtest_pred_df.index,
                        y=backtest_pred_df["close"],
                        mode="lines+markers",
                        name=f"{config['name']} (Backtest Pred)",
                        line=dict(width=2, dash="dot"),
                        marker=dict(size=5),
                        opacity=0.7,
                    )
                )

            # Get the actual test data from the first available backtest result
            # (all models should have same test dates since they're from same data)
            first_result = next(iter(backtest_results_all.values()))
            test_df = first_result["test_df"]

            fig_comparison.add_trace(
                go.Scatter(
                    x=test_df.index,
                    y=test_df["close"],
                    mode="lines+markers",
                    name="Actual (Backtest Period)",
                    line=dict(color="#FF6B6B", width=3),
                    marker=dict(size=8, symbol="diamond"),
                )
            )

        fig_comparison.update_layout(
            title="Price Predictions Comparison Across Models",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=600,
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        st.plotly_chart(fig_comparison, use_container_width=True)

        st.subheader("📈 Price Predictions Comparison (Candlestick)")
        fig_comparison_candle = go.Figure()

        fig_comparison_candle.add_trace(
            go.Candlestick(
                x=hist_df.index,
                open=hist_df["open"],
                high=hist_df["high"],
                low=hist_df["low"],
                close=hist_df["close"],
                name="Historical",
                increasing_line_color="green",
                decreasing_line_color="red",
                opacity=0.8,
            )
        )

        candle_colors = [
            ("blue", "orange"),
            ("cyan", "magenta"),
            ("yellow", "purple"),
            ("lime", "deeppink"),
        ]
        comparison_required_cols = ["open", "high", "low", "close"]

        for idx, (model_key, predictions) in enumerate(all_predictions.items()):
            pred_df = predictions["pred_df"]
            config = predictions["config"]

            for col in comparison_required_cols:
                if col not in pred_df.columns:
                    pred_df[col] = pred_df["close"]

            inc_color, dec_color = candle_colors[idx % len(candle_colors)]
            fig_comparison_candle.add_trace(
                go.Candlestick(
                    x=pred_df.index,
                    open=pred_df["open"],
                    high=pred_df["high"],
                    low=pred_df["low"],
                    close=pred_df["close"],
                    name=f"{config['name']} Prediction",
                    increasing_line_color=inc_color,
                    decreasing_line_color=dec_color,
                    opacity=0.6,
                )
            )

        if st.session_state.get("backtest_run_all", False) and st.session_state.get(
            "backtest_results_all"
        ):
            backtest_results_all = st.session_state.backtest_results_all

            for model_key, backtest_result in backtest_results_all.items():
                config = backtest_result["config"]
                backtest_pred_df = backtest_result["pred_df"]

                for col in comparison_required_cols:
                    if col not in backtest_pred_df.columns:
                        backtest_pred_df[col] = backtest_pred_df["close"]

                fig_comparison_candle.add_trace(
                    go.Candlestick(
                        x=backtest_pred_df.index,
                        open=backtest_pred_df["open"],
                        high=backtest_pred_df["high"],
                        low=backtest_pred_df["low"],
                        close=backtest_pred_df["close"],
                        name=f"{config['name']} (Backtest Pred)",
                        increasing_line_color="royalblue",
                        decreasing_line_color="darkorange",
                        opacity=0.5,
                    )
                )

            first_result = next(iter(backtest_results_all.values()))
            test_df = first_result["test_df"]

            fig_comparison_candle.add_trace(
                go.Scatter(
                    x=test_df.index,
                    y=test_df["close"],
                    mode="lines+markers",
                    name="Actual (Backtest Period)",
                    line=dict(color="#FF6B6B", width=3),
                    marker=dict(size=6, symbol="diamond"),
                    opacity=0.8,
                )
            )

        fig_comparison_candle.update_layout(
            title="Price Predictions Comparison Across Models (Candlestick)",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=700,
            xaxis_rangeslider_visible=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        st.plotly_chart(fig_comparison_candle, use_container_width=True)

    # ---- BACKTESTING SECTION ----
    st.markdown("---")
    st.subheader("🔍 Backtest Accuracy Check")

    # Backtest configuration
    col_backtest1, col_backtest2 = st.columns(2)
    with col_backtest1:
        backtest_pred_len = st.slider(
            "Backtest Period (days)", 5, 60, 30, key="backtest_period"
        )
    with col_backtest2:
        backtest_lookback = st.slider(
            "Backtest Lookback", 100, 512, 256, key="backtest_lookback"
        )

    if st.button(
        "🧪 Run Backtest for All Models",
        type="primary",
        help="Test prediction accuracy using historical data",
    ):
        with st.spinner("Running backtests..."):
            # Check if we have enough data
            if len(df) < backtest_pred_len + 50:
                st.error(
                    f"Not enough data for backtest. Need at least {backtest_pred_len + 50} rows, but only have {len(df)}."
                )
            else:
                backtest_results_all = {}

                # Run backtest for each selected model
                for model_key in selected_models:
                    predictor, config = available_predictors[model_key]

                    # Adjust lookback based on model's context length
                    model_backtest_lookback = min(
                        backtest_lookback, config["context_length"]
                    )

                    try:
                        pred_df, test_df = run_backtest(
                            df,
                            predictor,
                            lookback=min(
                                model_backtest_lookback, len(df) - backtest_pred_len
                            ),
                            pred_len=backtest_pred_len,
                        )

                        # Calculate metrics
                        backtest_metrics = calculate_backtest_metrics(pred_df, test_df)

                        backtest_results_all[model_key] = {
                            "pred_df": pred_df,
                            "test_df": test_df,
                            "metrics": backtest_metrics,
                            "pred_len": backtest_pred_len,
                            "config": config,
                        }
                    except Exception as e:
                        st.error(f"Backtest failed for {config['name']}: {str(e)}")
                        continue

                st.session_state.backtest_run_all = True
                st.session_state.backtest_results_all = backtest_results_all
                st.success(
                    f"✅ Backtests completed for {len(backtest_results_all)} model(s)!"
                )
                st.rerun()

    # Display backtest results
    if st.session_state.get("backtest_run_all", False) and st.session_state.get(
        "backtest_results_all"
    ):
        backtest_results_all = st.session_state.backtest_results_all

        # ===========================
        # CONSOLIDATED METRICS TABLE
        # ===========================
        st.subheader("📊 Backtest Metrics Comparison - All Models")

        # Build consolidated metrics dataframe
        metrics_comparison = []

        for model_key, results in backtest_results_all.items():
            config = results["config"]
            metrics = results["metrics"]
            pred_len = results["pred_len"]

            # Extract metrics for close price (primary metric)
            close_metrics = metrics.get("close", {})

            metrics_comparison.append(
                {
                    "Model": config["name"],
                    "Context Length": config["context_length"],
                    "Parameters": config["params"],
                    "MAE": close_metrics.get("MAE", 0),
                    "RMSE": close_metrics.get("RMSE", 0),
                    "MAPE (%)": close_metrics.get("MAPE (%)", 0),
                    "Data Points": close_metrics.get("Count", 0),
                    "Test Period": f"{pred_len} days",
                }
            )

        metrics_df = pd.DataFrame(metrics_comparison)

        # Display the consolidated table with highlighting and proper contrast
        def style_metrics_table(df):
            """Apply styling with proper text contrast for dark theme"""
            # Create a copy to avoid modifying original
            styled = df.style

            # Define color schemes with good contrast
            # Green background with dark text for best (min) values
            # Red background with white text for worst (max) values
            for col in ["MAE", "RMSE", "MAPE (%)"]:
                if col in df.columns:
                    # Apply styling for min values (best - green background, dark text)
                    styled = styled.map(
                        lambda x: (
                            "background-color: #90EE90; color: #000000; font-weight: bold"
                            if x == df[col].min()
                            else ""
                        ),
                        subset=[col],
                    )

                    # Apply styling for max values (worst - red background, white text)
                    styled = styled.map(
                        lambda x: (
                            "background-color: #FF6B6B; color: #FFFFFF; font-weight: bold"
                            if x == df[col].max()
                            else ""
                        ),
                        subset=[col],
                    )

            # Format the numeric columns for display
            styled = styled.format(
                {
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "MAPE (%)": "{:.2f}%",
                }
            )

            return styled

        st.dataframe(
            style_metrics_table(metrics_df),
            use_container_width=True,
            hide_index=True,
        )

        # Add explanation
        st.caption(
            "🟢 **Light Green (black text)** = Best (lowest error) | 🔴 **Red (white text)** = Worst (highest error) | "
            "Lower MAE, RMSE, and MAPE values indicate better prediction accuracy"
        )

        st.markdown("---")

        # ===========================
        # INDIVIDUAL MODEL TABS
        # ===========================
        # Create tabs for backtest results
        backtest_tab_list = [
            backtest_results_all[k]["config"]["name"]
            for k in backtest_results_all.keys()
        ]
        backtest_tabs = st.tabs(backtest_tab_list)

        for tab_idx, (model_key, results) in enumerate(backtest_results_all.items()):
            with backtest_tabs[tab_idx]:
                pred_df = results["pred_df"]
                test_df = results["test_df"]
                backtest_metrics = results["metrics"]
                config = results["config"]

                # Display metrics
                st.subheader(f"📊 Backtest Metrics - {config['name']}")

                for col_name, metrics in backtest_metrics.items():
                    with st.expander(
                        f"{col_name.upper()} - Accuracy Metrics", expanded=True
                    ):
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(
                            4
                        )

                        metric_col1.metric(
                            "MAE",
                            f"{metrics['MAE']:.4f}",
                            help="Mean Absolute Error - lower is better",
                        )
                        metric_col2.metric(
                            "RMSE",
                            f"{metrics['RMSE']:.4f}",
                            help="Root Mean Square Error - lower is better",
                        )
                        metric_col3.metric(
                            "MAPE (%)",
                            f"{metrics['MAPE (%)']:.2f}%",
                            help="Mean Absolute Percentage Error - lower is better",
                        )
                        metric_col4.metric(
                            "Data Points",
                            f"{metrics['Count']}",
                            help="Number of data points used in calculation",
                        )

                # Plot backtest results
                st.subheader(f"📈 Backtest: Actual vs Predicted - {config['name']}")
                fig_backtest = plot_backtest_results(test_df, pred_df)
                st.plotly_chart(fig_backtest, use_container_width=True)

                # Show data comparison table
                st.subheader("📋 Detailed Comparison")
                comparison_df = pd.DataFrame(
                    {
                        "Actual_Close": (
                            test_df["close"] if "close" in test_df.columns else None
                        ),
                        "Predicted_Close": (
                            pred_df["close"] if "close" in pred_df.columns else None
                        ),
                    }
                )
                comparison_df["Error"] = (
                    comparison_df["Actual_Close"] - comparison_df["Predicted_Close"]
                )
                comparison_df["Error_%"] = (
                    comparison_df["Error"] / comparison_df["Actual_Close"] * 100
                ).round(2)

                st.dataframe(
                    comparison_df.style.background_gradient(
                        subset=["Error_%"], cmap="RdYlGn", axis=0
                    )
                )

                # Volume comparison if available
                if "volume" in test_df.columns and "volume" in pred_df.columns:
                    st.subheader("📊 Volume: Actual vs Predicted")
                    fig_vol_backtest = go.Figure()

                    fig_vol_backtest.add_trace(
                        go.Bar(
                            x=test_df.index,
                            y=test_df["volume"],
                            name="Actual Volume",
                            marker_color="blue",
                            opacity=0.7,
                        )
                    )

                    fig_vol_backtest.add_trace(
                        go.Bar(
                            x=pred_df.index,
                            y=pred_df["volume"],
                            name="Predicted Volume",
                            marker_color="red",
                            opacity=0.7,
                        )
                    )

                    fig_vol_backtest.update_layout(
                        title=f"Backtest: Volume Comparison - {config['name']}",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        template="plotly_dark",
                        height=400,
                        barmode="group",
                    )

                    st.plotly_chart(fig_vol_backtest, use_container_width=True)

                st.info(
                    f"💡 Backtest results from the last run with {results['pred_len']} days prediction period."
                )
