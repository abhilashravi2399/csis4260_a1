import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

FEATURES_PATH = "data/features.parquet"
RESULTS_DIR = "results"
FEATURE_COLS = ["ret_1", "ret_5", "close_vs_sma", "close_vs_ema",
                "rsi_14", "hl_range", "vol_change"]


def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def mape(actual, pred):
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def build_dataset():
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values(["name", "date"]).reset_index(drop=True)

    # extra features that do not depend on the price level
    df["log_vol"] = np.log1p(df["volume"])
    df["ret_1"] = df.groupby("name")["close"].pct_change()
    df["ret_5"] = df.groupby("name")["close"].pct_change(5)
    df["vol_change"] = df.groupby("name")["log_vol"].diff()
    df["close_vs_sma"] = df["close"] / df["sma_20"] - 1
    df["close_vs_ema"] = df["close"] / df["ema_12"] - 1
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # target: the next day's close for the same company
    df["next_close"] = df.groupby("name")["close"].shift(-1)
    df["next_date"] = df.groupby("name")["date"].shift(-1)
    df["target_ret"] = df["next_close"] / df["close"] - 1

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["target_ret"]).reset_index(drop=True)

    # 80-20 split in time order, inside each company
    position = df.groupby("name").cumcount()
    size = df.groupby("name")["date"].transform("size")
    df["split"] = np.where(position < 0.8 * size, "train", "test")
    return df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = build_dataset()
    train = df[df["split"] == "train"]
    print("Train rows:", len(train), " Test rows:", (df["split"] == "test").sum())

    models = {
        "linear": LinearRegression(),
        "xgboost": XGBRegressor(n_estimators=300, max_depth=4,
                                learning_rate=0.05, subsample=0.8,
                                n_jobs=-1, random_state=42),
    }

    train_times = {"naive": 0.0}
    df["pred_naive"] = df["close"]

    for model_name, model in models.items():
        print("Training", model_name, "...")
        start = time.perf_counter()
        model.fit(train[FEATURE_COLS], train["target_ret"])
        train_times[model_name] = time.perf_counter() - start

        predicted_return = model.predict(df[FEATURE_COLS])
        df["pred_" + model_name] = df["close"] * (1 + predicted_return)

    # scores on the test part only
    test = df[df["split"] == "test"]
    model_names = ["naive", "linear", "xgboost"]

    summary = []
    for m in model_names:
        summary.append({
            "model": m,
            "rmse": round(rmse(test["next_close"], test["pred_" + m]), 3),
            "mae": round(mae(test["next_close"], test["pred_" + m]), 3),
            "mape_pct": round(mape(test["next_close"], test["pred_" + m]), 3),
            "train_s": round(train_times[m], 2),
        })
    summary_df = pd.DataFrame(summary)
    table_text = summary_df.to_markdown(index=False)
    print()
    print(table_text)

    with open(f"{RESULTS_DIR}/part2_model_results.md", "w") as f:
        f.write(table_text)

    # scores for each company, used by the dashboard
    rows = []
    for ticker, part in test.groupby("name"):
        row = {"name": ticker}
        for m in model_names:
            row[m + "_rmse"] = rmse(part["next_close"], part["pred_" + m])
            row[m + "_mae"] = mae(part["next_close"], part["pred_" + m])
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/part2_metrics_by_ticker.csv",
                              index=False)

    # predictions for every company, used by the dashboard
    out = df[["name", "next_date", "next_close", "split",
              "pred_naive", "pred_linear", "pred_xgboost"]].copy()
    out = out.rename(columns={"next_date": "date", "next_close": "actual_close"})
    out.to_parquet("data/predictions.parquet", index=False)

    print("\nSaved data/predictions.parquet and the results files.")


if __name__ == "__main__":
    main()