import os
import time

import numpy as np
import pandas as pd
import polars as pl

CSV_PATH = "data/all_stocks_5yr.csv"
RESULTS_DIR = "results"
REPEATS = 5


def median_time(func, repeats):
    # runs the function a few times and returns the median in seconds
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


# ---------- pandas ----------
def pandas_load():
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


def pandas_features(df):
    df = df.sort_values(["name", "date"]).reset_index(drop=True)
    grp = df.groupby("name")["close"]

    df["sma_20"] = grp.transform(lambda s: s.rolling(20).mean())
    df["ema_12"] = grp.transform(lambda s: s.ewm(span=12, adjust=False).mean())

    delta = grp.diff()
    df["gain"] = delta.clip(lower=0)
    df["loss"] = (-delta).clip(lower=0)
    avg_gain = df.groupby("name")["gain"].transform(lambda s: s.rolling(14).mean())
    avg_loss = df.groupby("name")["loss"].transform(lambda s: s.rolling(14).mean())
    df["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss)

    return df.drop(columns=["gain", "loss"])


# ---------- polars ----------
def polars_load():
    return pl.read_csv(CSV_PATH, try_parse_dates=True)


def polars_features(df):
    df = df.sort(["name", "date"])
    df = df.with_columns(
        pl.col("close").rolling_mean(20).over("name").alias("sma_20"),
        pl.col("close").ewm_mean(span=12, adjust=False).over("name").alias("ema_12"),
        pl.col("close").diff().over("name").alias("delta"),
    )
    df = df.with_columns(
        pl.col("delta").clip(lower_bound=0).alias("gain"),
        (-pl.col("delta")).clip(lower_bound=0).alias("loss"),
    )
    df = df.with_columns(
        pl.col("gain").rolling_mean(14).over("name").alias("avg_gain"),
        pl.col("loss").rolling_mean(14).over("name").alias("avg_loss"),
    )
    df = df.with_columns(
        (100 - 100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))).alias("rsi_14")
    )
    return df.drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pdf = pandas_load()
    pldf = polars_load()

    print("Timing the load step...")
    pandas_load_t = median_time(pandas_load, REPEATS)
    polars_load_t = median_time(polars_load, REPEATS)

    print("Timing the indicator step...")
    pandas_feat_t = median_time(lambda: pandas_features(pdf), REPEATS)
    polars_feat_t = median_time(lambda: polars_features(pldf), REPEATS)

    # check that both libraries give the same numbers
    p = pandas_features(pdf)
    q = polars_features(pldf)
    print("\nMax difference between pandas and polars:")
    for col in ["sma_20", "ema_12", "rsi_14"]:
        diff = np.nanmax(np.abs(p[col].to_numpy() - q[col].to_numpy()))
        print(" ", col, diff)

    results = pd.DataFrame({
        "step": ["load csv", "indicators", "total"],
        "pandas_s": [pandas_load_t, pandas_feat_t, pandas_load_t + pandas_feat_t],
        "polars_s": [polars_load_t, polars_feat_t, polars_load_t + polars_feat_t],
    })
    results["pandas_s"] = results["pandas_s"].round(3)
    results["polars_s"] = results["polars_s"].round(3)
    results["polars_speedup"] = (results["pandas_s"] / results["polars_s"]).round(1)

    table_text = results.to_markdown(index=False)
    print()
    print(table_text)

    with open(f"{RESULTS_DIR}/part2_features_timing.md", "w") as f:
        f.write(table_text)

    # save the features for the model step
    q.write_parquet("data/features.parquet")
    print("\nSaved data/features.parquet and the timing table.")


if __name__ == "__main__":
    main()