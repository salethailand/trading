import pandas as pd
src = "btcusd_1-min_data_2_clean.csv"
df  = pd.read_csv(src)
df_dupes = pd.concat([df, df.iloc[100:110], df.iloc[500:505]], ignore_index=True)
df_dupes.to_csv("btcusd_1-min_data_2_clean_DUPES.csv", index=False)
print("Wrote duped file with", len(df_dupes)-len(df), "extra rows")
