import os
import time
import gc
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# settings you can change
CSV_PATH = "data/all_stocks_5yr.csv"
SCALED_DIR = "C:/temp_scaled"
RESULTS_DIR = "results"

SCALES = [1, 10]                      # run 100 later, separately
REPEATS = {1: 5, 10: 5, 100: 1}       # how many times each read is timed
COMPRESSIONS = ["snappy", "gzip", "zstd"]


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def write_csv_scaled(df, path, scale):
    # writes the same data "scale" times into one csv file
    start = time.perf_counter()
    for i in range(scale):
        if i == 0:
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False, mode="a", header=False)
    return time.perf_counter() - start


def write_parquet_scaled(table, path, scale, compression):
    # writes the same data "scale" times into one parquet file
    start = time.perf_counter()
    writer = pq.ParquetWriter(path, table.schema, compression=compression)
    for i in range(scale):
        writer.write_table(table)
    writer.close()
    return time.perf_counter() - start


def time_read(read_function, repeats):
    # one untimed warm-up read, then the median of the timed reads
    times = []
    try:
        if repeats > 1:
            data = read_function()
            del data
            gc.collect()
        for i in range(repeats):
            start = time.perf_counter()
            data = read_function()
            times.append(time.perf_counter() - start)
            del data
            gc.collect()
    except MemoryError:
        return None
    times.sort()
    return times[len(times) // 2]


def main():
    os.makedirs(SCALED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading the original csv...")
    df = pd.read_csv(CSV_PATH)
    table = pa.Table.from_pandas(df, preserve_index=False)
    print("Rows in 1x data:", len(df))
    print("Columns found:", df.columns.tolist())
    ticker_col = df.columns[-1]          # the ticker column is the last one
    subset_cols = ["date", "close", ticker_col]

    formats = [("csv", None)]
    for c in COMPRESSIONS:
        formats.append(("parquet-" + c, c))

    rows = []

    for scale in SCALES:
        reps = REPEATS[scale]

        for name, comp in formats:
            print(f"Scale {scale}x, format {name} ...")

            if comp is None:
                path = f"{SCALED_DIR}/stocks_{scale}x.csv"
                write_time = write_csv_scaled(df, path, scale)
                read_full = lambda p=path: pd.read_csv(p)
                read_part = lambda p=path: pd.read_csv(p, usecols=subset_cols)
            else:
                path = f"{SCALED_DIR}/stocks_{scale}x_{comp}.parquet"
                write_time = write_parquet_scaled(table, path, scale, comp)
                read_full = lambda p=path: pd.read_parquet(p)
                read_part = lambda p=path: pd.read_parquet(p, columns=subset_cols)

            size = file_size_mb(path)
            full_time = time_read(read_full, reps)
            part_time = time_read(read_part, reps)

            rows.append({
                "scale": f"{scale}x",
                "format": name,
                "size_mb": round(size, 1),
                "write_s": round(write_time, 2),
                "full_read_s": None if full_time is None else round(full_time, 2),
                "3col_read_s": None if part_time is None else round(part_time, 2),
            })

            os.remove(path)   # delete the big file straight away

    results = pd.DataFrame(rows)
    table_text = results.to_markdown(index=False)
    print()
    print(table_text)

    tag = "_".join(str(s) + "x" for s in SCALES)
    with open(f"{RESULTS_DIR}/part1_results_{tag}.md", "w") as f:
        f.write(table_text)
    results.to_csv(f"{RESULTS_DIR}/part1_results_{tag}.csv", index=False)

    shutil.rmtree(SCALED_DIR, ignore_errors=True)
    print("\nSaved the results in the results folder.")


if __name__ == "__main__":
    main()