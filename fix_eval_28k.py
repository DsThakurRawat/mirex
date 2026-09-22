import pandas as pd
import numpy as np

csv_p = r"d:\mirex\Architecture-1\scripts\eval_28k_results.csv"
df = pd.read_csv(csv_p)
print("Total rows:", len(df))
print("NaN score count:", df["score"].isna().sum())
print("Null rows:")
print(df[df["score"].isna()][["file_path", "strat_class", "is_ai"]])

# Fill NaN scores with default fallback score 0.5 (or impute)
df["score"] = df["score"].fillna(0.5)
df.to_csv(csv_p, index=False)
print("Cleaned and saved CSV!")
