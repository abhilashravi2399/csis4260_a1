from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "predictions.parquet"
MODEL_LABELS = {
    "linear": "Linear regression",
    "xgboost": "XGBoost",
    "naive": "Naive (previous close)",
}

st.set_page_config(page_title="Stock price predictions", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["name", "date"]).reset_index(drop=True)


def score(actual, pred):
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return rmse, mae, mape


df = load_data()
tickers = sorted(df["name"].unique())

st.title("Next-day closing price predictions")
st.caption("S&P 500 daily data, 2013-02-08 to 2018-02-07. "
           "Models were trained on the first 80% of each company's days "
           "and tested on the last 20%.")

# ---------- sidebar ----------
st.sidebar.header("Controls")
default_index = tickers.index("AAPL") if "AAPL" in tickers else 0
ticker = st.sidebar.selectbox("Company ticker (click and type to search)",
                              tickers, index=default_index)

chosen_models = st.sidebar.multiselect(
    "Models to show",
    options=list(MODEL_LABELS.keys()),
    default=["linear", "xgboost"],
    format_func=lambda key: MODEL_LABELS[key],
)
include_train = st.sidebar.checkbox("Include the training period", value=False)

part = df[df["name"] == ticker]
if not include_train:
    part = part[part["split"] == "test"]

first_day = part["date"].min().date()
last_day = part["date"].max().date()
date_range = st.sidebar.slider("Date range", min_value=first_day,
                               max_value=last_day, value=(first_day, last_day))

part = part[(part["date"].dt.date >= date_range[0]) &
            (part["date"].dt.date <= date_range[1])]

# ---------- metrics ----------
st.subheader(f"{ticker}: accuracy on the test period")
test_part = part[part["split"] == "test"]

if len(test_part) == 0:
    st.warning("There are no test days in the chosen date range.")
elif len(chosen_models) == 0:
    st.info("Pick at least one model in the sidebar.")
else:
    columns = st.columns(len(chosen_models))
    for column, model in zip(columns, chosen_models):
        rmse, mae, mape = score(test_part["actual_close"], test_part["pred_" + model])
        with column:
            st.markdown(f"**{MODEL_LABELS[model]}**")
            st.metric("RMSE", f"{rmse:.3f}")
            st.metric("MAE", f"{mae:.3f}")
            st.metric("MAPE", f"{mape:.2f}%")

# ---------- price chart ----------
st.subheader(f"{ticker}: actual vs predicted closing price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=part["date"], y=part["actual_close"],
                         name="Actual", line=dict(width=2)))
for model in chosen_models:
    fig.add_trace(go.Scatter(x=part["date"], y=part["pred_" + model],
                             name=MODEL_LABELS[model], line=dict(dash="dot")))

if include_train:
    split_date = df[(df["name"] == ticker) & (df["split"] == "test")]["date"].min()
    fig.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1,
                  yref="paper", line=dict(dash="dash"))
    fig.add_annotation(x=split_date, y=1, yref="paper", text="test period starts",
                       showarrow=False, xanchor="left")

fig.update_layout(xaxis_title="Date", yaxis_title="Closing price (USD)",
                  hovermode="x unified", height=450,
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

# ---------- error chart ----------
st.subheader(f"{ticker}: prediction error (predicted minus actual)")
fig2 = go.Figure()
for model in chosen_models:
    error = part["pred_" + model] - part["actual_close"]
    fig2.add_trace(go.Scatter(x=part["date"], y=error,
                              name=MODEL_LABELS[model], mode="lines"))
fig2.update_layout(xaxis_title="Date", yaxis_title="Error (USD)",
                   hovermode="x unified", height=300,
                   legend=dict(orientation="h", y=-0.3))
st.plotly_chart(fig2, use_container_width=True)

# ---------- data table ----------
with st.expander("Show the last 20 rows of data"):
    st.dataframe(part.tail(20), use_container_width=True)