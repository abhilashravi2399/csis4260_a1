import pandas as pd

df = pd.read_csv("data/all_stocks_5yr.csv")
print(df.shape)
print(df.head())
print(df.dtypes)
print(df["Name"].nunique())