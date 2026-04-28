import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Market Pattern Intelligence")

st.markdown(
    """
<style>
.metric-card {
    background-color: #0f172a;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #1e293b;
}
.big-font {
    font-size: 20px;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 Market Behavior & Pattern Intelligence")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload OHLCV CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload CSV with: timestamp, open, high, low, close, volume")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = [c.lower() for c in df.columns]

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

df = df.sort_index()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Parameters")

vol_window = st.sidebar.slider("Volatility Window", 5, 50, 20)
trend_window = st.sidebar.slider("Trend Window", 10, 200, 50)
swing_window = st.sidebar.slider("Swing Window", 2, 20, 5)

# =========================
# METRICS
# =========================
df["return"] = df["close"].pct_change()
df["range_pct"] = (df["high"] - df["low"]) / df["close"]

df["volatility"] = df["return"].rolling(vol_window).std()
vol_threshold = df["volatility"].quantile(0.7)
df["high_vol"] = df["volatility"] > vol_threshold

df["ma"] = df["close"].rolling(trend_window).mean()
df["trend_up"] = df["close"] > df["ma"]

df["roll_max"] = df["high"].rolling(swing_window).max()
df["roll_min"] = df["low"].rolling(swing_window).min()

df["up_move"] = (df["roll_max"] - df["close"]) / df["close"]
df["down_move"] = (df["roll_min"] - df["close"]) / df["close"]

# Convert for plotting
df["return_pct"] = df["return"] * 100
df["range_pct_100"] = df["range_pct"] * 100
df["up_move_pct"] = df["up_move"] * 100
df["down_move_pct"] = df["down_move"] * 100

# =========================
# KPI CARDS
# =========================
col1, col2, col3 = st.columns(3)

avg_vol = df["volatility"].mean()
avg_range = df["range_pct"].mean()
trend_strength = df["trend_up"].mean()

col1.markdown(
    f"<div class='metric-card'><div>Avg Volatility</div><div class='big-font'>{avg_vol:.2%}</div></div>",
    unsafe_allow_html=True,
)
col2.markdown(
    f"<div class='metric-card'><div>Avg Range</div><div class='big-font'>{avg_range:.2%}</div></div>",
    unsafe_allow_html=True,
)
col3.markdown(
    f"<div class='metric-card'><div>Trend Presence</div><div class='big-font'>{trend_strength:.2%}</div></div>",
    unsafe_allow_html=True,
)

# =========================
# DISTRIBUTIONS
# =========================
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="return_pct", nbins=100, title="Daily Returns (%)")
    fig.update_traces(hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df, x="range_pct_100", nbins=100, title="Intraday Range (%)")
    fig.update_traces(hovertemplate="Range: %{x:.2f}%<br>Count: %{y}<extra></extra>")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# CDF + EXCEEDANCE
# =========================
st.subheader("📈 CDF & Exceedance Probability")

returns = df["return_pct"].dropna()
sorted_r = np.sort(returns)

cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
exceedance = 1 - cdf

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=sorted_r,
        y=cdf,
        mode="lines",
        name="CDF",
        hovertemplate="Return: %{x:.2f}%<br>CDF: %{y:.3f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=sorted_r,
        y=exceedance,
        mode="lines",
        name="Exceedance",
        hovertemplate="Return: %{x:.2f}%<br>Exceedance: %{y:.3f}<extra></extra>",
    )
)

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Return (%)",
    yaxis_title="Probability",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# SWING SCATTER
# =========================
fig = px.scatter(df, x="down_move_pct", y="up_move_pct", opacity=0.4)

fig.update_traces(
    customdata=np.stack([df["volatility"]], axis=-1),
    hovertemplate=(
        "Adverse: %{x:.2f}%<br>"
        "Favorable: %{y:.2f}%<br>"
        "Volatility: %{customdata[0]:.4f}<extra></extra>"
    ),
)

fig.update_layout(
    template="plotly_dark",
    title="Adverse vs Favorable Moves",
    xaxis_title="Adverse (%)",
    yaxis_title="Favorable (%)",
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# REGIME COMPARISON
# =========================
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df[df["high_vol"]], x="return_pct", title="High Volatility")
    fig.update_traces(hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df[~df["high_vol"]], x="return_pct", title="Low Volatility")
    fig.update_traces(hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# STOP DISTRIBUTION
# =========================
st.subheader("🎯 Adverse Move Distribution")

down = df["down_move"].dropna()

p70 = abs(down.quantile(0.7)) * 100
p85 = abs(down.quantile(0.85)) * 100
p95 = abs(down.quantile(0.95)) * 100

stop_df = pd.DataFrame(
    {"Type": ["70th %ile", "85th %ile", "95th %ile"], "Move (%)": [p70, p85, p95]}
)

fig = px.bar(
    stop_df, x="Type", y="Move (%)", text=stop_df["Move (%)"].map(lambda x: f"{x:.1f}%")
)

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# =========================
# REGIME MOVES
# =========================
st.subheader("⚡ Regime-Based Behavior")

high_vol_down = df[df["high_vol"]]["down_move"].dropna()
low_vol_down = df[~df["high_vol"]]["down_move"].dropna()

hv = abs(high_vol_down.quantile(0.85)) * 100
lv = abs(low_vol_down.quantile(0.85)) * 100

st.markdown(
    f"""
- High Volatility typical adverse move: **{hv:.2f}%**
- Low Volatility typical adverse move: **{lv:.2f}%**
"""
)

# =========================
# PATTERN INSIGHTS
# =========================
st.subheader("🧠 Pattern Inference")

insight = f"""
### Market Behavior Summary:

- Most price pullbacks stay within **{p70:.2f}%**
- Majority of extreme moves are capped near **{p95:.2f}%**
- Typical adverse movement cluster around **{p85:.2f}%**
- High volatility regimes expand adverse moves to **{hv:.2f}%**
- Low volatility compresses movement to **{lv:.2f}%**

### Structural Observations:

- If adverse vs favorable scatter is symmetric → market is **mean-reverting**
- If favorable moves dominate → **trend persistence exists**
- Wide dispersion → **unstable / news-driven market**
- Tight clustering → **controlled / algorithmic behavior**

### Regime Interpretation:

- High volatility → larger swings, less predictable structure  
- Low volatility → tighter, more repeatable patterns  
- Strong trend presence → directional bias dominates pullbacks  

"""


st.markdown(insight)

from sklearn.cluster import KMeans

st.subheader("🧠 Advanced Regime Detection")

# =========================
# PREP DATA FOR CLUSTERING
# =========================
cluster_df = (
    df[["return", "volatility", "range_pct", "up_move", "down_move"]].dropna().copy()
)

# Normalize (important)
cluster_scaled = (cluster_df - cluster_df.mean()) / cluster_df.std()

# =========================
# KMEANS CLUSTERING
# =========================
k = st.slider("Number of Regimes", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_df["regime"] = kmeans.fit_predict(cluster_scaled)

df.loc[cluster_df.index, "regime"] = cluster_df["regime"]

# =========================
# REGIME VISUALIZATION (SCATTER)
# =========================
fig = px.scatter(
    cluster_df,
    x=cluster_df["down_move"] * 100,
    y=cluster_df["up_move"] * 100,
    color=cluster_df["regime"].astype(str),
    title="Regime Clusters (Adverse vs Favorable %)",
    opacity=0.5,
)

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Adverse Move (%)",
    yaxis_title="Favorable Move (%)",
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DENSITY HEATMAP
# =========================
st.subheader("🌡️ Behavior Density Map")

fig = px.density_heatmap(
    df,
    x="down_move_pct",
    y="up_move_pct",
    nbinsx=60,
    nbinsy=60,
    title="Where Price Behavior Concentrates",
)

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# =========================
# PERCENTILE BANDS OVERLAY
# =========================
st.subheader("📊 Percentile Structure")

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df["down_move_pct"], y=df["up_move_pct"], mode="markers", opacity=0.2)
)

# Add percentile lines
for val, name in zip([p70, p85, p95], ["70%", "85%", "95%"]):
    fig.add_vline(x=val, line_dash="dash", annotation_text=name)
    fig.add_hline(y=val, line_dash="dash")

fig.update_layout(
    template="plotly_dark",
    title="Percentile Zones (Adverse vs Favorable)",
    xaxis_title="Adverse Move (%)",
    yaxis_title="Favorable Move (%)",
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# REGIME TIMELINE
# =========================
st.subheader("⏱️ Regime Timeline")

timeline_df = df.dropna(subset=["regime"]).copy()

fig = px.scatter(
    timeline_df,
    x=timeline_df.index,
    y=[1] * len(timeline_df),
    color=timeline_df["regime"].astype(str),
    title="Market Regime Over Time",
)

fig.update_layout(template="plotly_dark", yaxis_visible=False)

st.plotly_chart(fig, use_container_width=True)

# =========================
# REGIME SUMMARY
# =========================
st.subheader("🧾 Regime Intelligence Summary")

summary_text = ""

for r in sorted(cluster_df["regime"].unique()):
    subset = cluster_df[cluster_df["regime"] == r]

    avg_down = abs(subset["down_move"].mean()) * 100
    avg_up = abs(subset["up_move"].mean()) * 100
    avg_vol = subset["volatility"].mean() * 100

    behavior = ""

    if avg_vol > df["volatility"].mean():
        behavior += "High Volatility, "
    else:
        behavior += "Low Volatility, "

    if avg_up > avg_down:
        behavior += "Trend-Favoring"
    else:
        behavior += "Mean-Reverting"

    summary_text += f"""
**Regime {r}:**
- Avg Adverse Move: {avg_down:.2f}%
- Avg Favorable Move: {avg_up:.2f}%
- Volatility Level: {avg_vol:.2f}%
- Behavior: {behavior}

"""

st.markdown(summary_text)
