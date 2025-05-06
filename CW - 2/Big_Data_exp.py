import pandas as pd

df_base = pd.read_csv("/content/Airbnb_clean.csv")

big_df = pd.concat([df_base]*4, ignore_index=True).sample(
             frac=1, random_state=42).reset_index(drop=True)
big_df.to_csv("/content/Airbnb_big.csv", index=False)
print("Rows in big_df:", len(big_df))


# !pip install pyspark --quiet           # Colab one‑off
import time, psutil, pandas as pd
from pyspark.sql import SparkSession, functions as F

csv_path = "/content/Airbnb_big.csv"     # adjust if needed
GROUP_COL = "host_id"                    # change to neighbourhood_cleansed etc.

spark = (SparkSession.builder.appName("bench").master("local[*]").getOrCreate())
df_sp  = spark.read.option("header","true").option("inferSchema","true").csv(csv_path)

# -- Spark timing
t0 = time.perf_counter()
(df_sp.groupBy(GROUP_COL)
     .agg(F.mean("price"), F.count("*"))
     .count())
spark_time = time.perf_counter() - t0

# -- Pandas timing
t1 = time.perf_counter()
df_pd = pd.read_csv(csv_path)
(pd.DataFrame({"mean_price": df_pd.groupby(GROUP_COL)["price"].mean(),
               "listings": df_pd.groupby(GROUP_COL).size()}))
pandas_time = time.perf_counter() - t1

print(f"Spark:  {spark_time:.2f}s  |  Pandas: {pandas_time:.2f}s")
print(f"RAM after Pandas: {psutil.Process().memory_info().rss/1e9:.2f} GB")

import matplotlib.pyplot as plt

times = [spark_time, pandas_time]
labels = ["Spark (local[*])", "Pandas (single‑thread)"]

fig, ax = plt.subplots()
bars = ax.bar(labels, times, color=["#4C72B0", "#55A868"])
ax.set_ylabel("Wall‑clock seconds")
ax.set_title("Aggregation Runtime: Spark vs Pandas")
for b,t in zip(bars, times):
    ax.text(b.get_x()+b.get_width()/2, t+0.3, f"{t:.2f}s",
            ha="center", va="bottom")
plt.show(); fig.savefig("runtime_compare.png", dpi=300)
