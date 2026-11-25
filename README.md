# CSIS 4260 Assignment 1: S&P 500 stock prices

Name: Abhilash Ravindranathan
Student ID- 300414478
Dataset: daily prices for 505 S&P 500 companies, 2013-02-08 to 2018-02-07 (619,040 rows)

This project has three parts: (1) storing and retrieving the data, (2) calculating indicators and predicting the next-day closing price, and (3) a dashboard to view the predictions. Everything is written as plain Python scripts and run from VS Code.

---

## 1. How to run the code

**Requirements:** Python 3.12.0 (tested on Windows 11). Python 3.15 was also installed on my laptop, but several libraries do not publish installers for it yet, so I used 3.12.

**Setup (run from the project root folder):**

```
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Mac or Linux, use `python3.12 -m venv venv` and `source venv/bin/activate`.

**Data:** put `all_stocks_5yr.csv` inside a folder called `data/`. The CSV is not included in the submission.

**Run order:**

| Step | Command | What it does |
|---|---|---|
| Part 1 | `python part1_storage/benchmark_storage.py` | CSV vs Parquet benchmark. Edit `SCALES` near the top of the file to choose 1x, 10x or 100x. |
| Part 2a | `python part2_models/features_compare.py` | Calculates indicators in Pandas and Polars, times both, saves `data/features.parquet`. |
| Part 2b | `python part2_models/train_models.py` | Trains the models, saves `data/predictions.parquet` and the score tables. |
| Part 3 | `streamlit run part3_dashboard/app.py` | Opens the dashboard at http://localhost:8501. |

Part 2b needs the output of Part 2a, and Part 3 needs the output of Part 2b. The submission already contains `data/predictions.parquet`, so the dashboard can be started straight away.

**Notes on the scaled data (Part 1):** the 10x and 100x files are created by the script by writing the original data 10 and 100 times. They are written to `C:/temp_scaled` (outside the project, so OneDrive does not sync them) and deleted straight after each measurement. No 10x or 100x files are included in this submission.

---

## 2. Project structure

```
csis4260_a1/
├── README.md
├── requirements.txt
├── screenshots.pdf
├── part1_storage/benchmark_storage.py
├── part2_models/features_compare.py
├── part2_models/train_models.py
├── part3_dashboard/app.py
├── results/               (result tables as .md and .csv)
└── data/predictions.parquet
```

---

## 3. Part 1: Storing and retrieving data

### Method

I compared CSV with Parquet using three compression schemes: snappy, gzip and zstd. For each format and each scale (1x, 10x, 100x) the script measures:

- file size on disk
- time to write the file
- time to read the full file into Pandas
- time to read only three columns (`date`, `close`, `name`), which is a common pattern in analysis

Reads at 1x and 10x are repeated five times after one warm-up read, and the median is reported. This is because the first read of a file is often slower or faster for reasons unrelated to the format (the operating system cache, library start-up). At 100x each read is done once, because repeating it would take very long and use a lot of memory.

### Results

| scale | format         | size_mb | write_s | full_read_s | 3col_read_s |
|:------|:---------------|--------:|--------:|------------:|------------:|
| 1x    | csv            |    28.8 |    1.38 |        0.06 |        0.06 |
| 1x    | parquet-snappy |    10.0 |    0.27 |        0.30 |        0.02 |
| 1x    | parquet-gzip   |     8.3 |    4.11 |        0.04 |        0.02 |
| 1x    | parquet-zstd   |     8.3 |    0.18 |        0.03 |        0.02 |
| 10x   | csv            |   288.0 |   16.05 |        0.37 |        0.60 |
| 10x   | parquet-snappy |   100.3 |    2.28 |        0.30 |        0.22 |
| 10x   | parquet-gzip   |    82.7 |   43.58 |        0.17 |        0.11 |
| 10x   | parquet-zstd   |    83.2 |    2.49 |        0.16 |        0.12 |
| 100x  | csv            |     TBD |     TBD |         TBD |         TBD |
| 100x  | parquet-snappy |     TBD |     TBD |         TBD |         TBD |
| 100x  | parquet-gzip   |     TBD |     TBD |         TBD |         TBD |
| 100x  | parquet-zstd   |     TBD |     TBD |         TBD |         TBD |

The raw tables are saved in the `results/` folder.

### Analysis

**Size.** Parquet is about 3x smaller than CSV at every scale (28.8 MB against 8.3 to 10.0 MB at 1x, and 288 MB against 83 to 100 MB at 10x). Parquet stores each column together and compresses it, and stock data has many repeated and similar values, so it compresses well. gzip and zstd give almost the same size, and both are about 17% smaller than snappy.

**Write time.** gzip is very slow to write: 43.58 s at 10x, about 17 times slower than zstd, for almost no size benefit. zstd writes as quickly as snappy (2.49 s against 2.28 s at 10x) while compressing like gzip.

**Read time.** When only three columns are needed, Parquet is clearly faster at 10x (0.11 to 0.22 s against 0.60 s for CSV), because Parquet can skip the columns it does not need while a CSV reader has to go through every line. For full reads, the differences at 1x are tiny, under a fraction of a second.

**Limitation.** The scaled files are made by repeating the same 619,040 rows. This gives the right file size and row count, but real data growth would have new values, so compression ratios might differ slightly.

### Recommendation

| Scale | Recommended format | Reason |
|---|---|---|
| 1x | Parquet with zstd (CSV is also fine) | Every operation takes well under a second, so speed does not matter. Parquet is still 3x smaller and keeps column types, but CSV is simpler to open and inspect. |
| 10x | Parquet with zstd | About 3.5x smaller, much faster for column reads, and writes quickly. |
| 100x | TBD (expected: Parquet with zstd, to be confirmed by the 100x results above) | TBD |

gzip is not recommended at any scale because of its write time, and snappy is only worth choosing if the smallest possible write time matters more than disk space.

---

## 4. Part 2: Indicators, Pandas vs Polars, and models

### Indicators

Three technical indicators were calculated separately for each company and added to the dataframe:

- **SMA 20:** simple moving average of the closing price over 20 days.
- **EMA 12:** exponential moving average of the closing price (span 12), which weights recent days more.
- **RSI 14:** relative strength index, calculated from the average gain and average loss over 14 days, ranging from 0 to 100.

### Pandas vs Polars

Both libraries calculate the same indicators in `features_compare.py`. Before comparing speed, the script checks that the results match. The largest differences were 0.0 for SMA and RSI and about 7e-13 for EMA, which is floating-point rounding, so the two libraries give the same answer.

| step       | pandas_s | polars_s | polars_speedup |
|:-----------|---------:|---------:|---------------:|
| load csv   |    0.147 |    0.029 |            5.1 |
| indicators |    0.375 |    0.067 |            5.6 |
| total      |    0.521 |    0.095 |            5.5 |

**Choice.** Polars is about 5.5 times faster for both loading and the indicator calculations, and its group-wise window functions (`.over("name")`) are shorter to write than the Pandas `groupby().transform(lambda ...)` version. At the current 1x size, both finish in under a second, so the speed gain is small in practice. It would matter much more at 10x or 100x, where the Pandas lambda-based group operations would take proportionally longer. I used Polars for the feature step and then passed the result as a Parquet file to Pandas, because scikit-learn, XGBoost, and Streamlit work directly with Pandas dataframes.

### Models

Two algorithms were trained, plus a naive baseline:

- **Linear regression:** simple and fast.
- **XGBoost:** gradient boosted trees (300 trees, depth 4, learning rate 0.05).
- **Naive baseline:** predicts that tomorrow's close equals today's close.

**Features:** 1-day return, 5-day return, close relative to SMA 20, close relative to EMA 12, RSI 14, high-low range as a fraction of close, and the change in log volume.

**Target:** the next day's percentage return, which is then converted back to a closing price (`close * (1 + predicted return)`). I did this because tree models such as XGBoost cannot predict prices outside the range they saw in training, and prices rose over the period, so predicting raw prices would fail on the test years.

**Split:** for each company, the first 80% of its days are used for training and the last 20% for testing. The split is in time order with no shuffling, so the models never see the future (back testing). One model is trained on all 505 companies together.

### Results (test period, all companies)

| model   |  rmse |   mae | mape_pct | train_s |
|:--------|------:|------:|---------:|--------:|
| naive   | 2.247 | 0.913 |    0.944 |    0.00 |
| linear  | 2.246 | 0.912 |    0.943 |    0.38 |
| xgboost | 2.245 | 0.914 |    0.945 |    4.62 |

Train rows: 487,540. Test rows: 121,396.

### Analysis

The three results are practically identical, differing only in the third decimal place. Neither model beats the naive baseline by a meaningful amount. This is not a coding error. Next-day stock prices behave almost like a random walk, so today's closing price already contains nearly all the information that simple price-based indicators can offer. The honest conclusion is that these indicators do not give useful extra predictive power for one-day-ahead prices.

Since accuracy is the same, linear regression is the sensible choice: it trained in 0.38 s against 4.62 s for XGBoost. The dashboard shows both models and the naive baseline, so the comparison can be seen for every company.

Possible improvements, which I did not do: add market-wide or sector features, predict direction (up or down) instead of price, or train a separate model per company.

---

## 5. Part 3: Dashboard

### Library choice

I looked at three options:

- **Streamlit:** pure Python, no HTML or callback code, built-in widgets such as dropdowns and sliders, and caching with `@st.cache_data`. The whole script re-runs on each interaction, which is simple to understand.
- **Dash:** more control over layout and interactions through callbacks, but it needs more code for the same result.
- **Reflex:** builds full web applications in Python with a state model, which is more than this assignment needs and is a newer, smaller library.

I chose **Streamlit** because the requirement is a ticker selector that updates charts, which Streamlit does with very little code. Charts are drawn with Plotly so they are interactive (hover, zoom).

### Features

- Ticker dropdown that filters as you type, covering all 505 companies
- Actual vs predicted closing price for the selected company
- Prediction error chart (predicted minus actual)
- RMSE, MAE and MAPE for each model on the selected company's test period
- Sidebar controls: choose which models to show, date range slider, and a tickbox to include the training period (a marker shows where the test period starts)
- Data table of the latest rows

Screenshots are in `screenshots.pdf`.

---

## 6. Summary of decisions

| Part | Decision | Main reason |
|---|---|---|
| 1 | Parquet with zstd (final recommendation at 100x to be confirmed) | About 3x smaller than CSV, fast column reads, fast writes |
| 2 | Polars for indicators, Pandas for modelling | Polars is about 5.5x faster with identical results; the modelling libraries expect Pandas |
| 2 | Linear regression preferred over XGBoost | Same accuracy as XGBoost, and it trains faster |
| 3 | Streamlit with Plotly | Least code for the required ticker selector |
