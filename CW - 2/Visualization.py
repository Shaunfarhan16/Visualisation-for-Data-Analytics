#   Airbnb London  –  Section 3 Visual‑Analytics  (Leak‑free, 10 figures)

import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import geopandas as gpd, contextily as cx, warnings
from wordcloud import WordCloud

plt.rcParams.update({"figure.dpi":300, "figure.figsize":(7,4),
                     "font.family":"DejaVu Sans"})
sns.set_style("whitegrid"); warnings.filterwarnings("ignore")

CSV = r"C:\Visualization of Data Analytics\Airbnb_clean.csv"                         # <-- adjust if needed
df  = pd.read_csv(CSV)

# 0.  LEAKAGE FIX
leak_cols = [c for c in df.columns if "price" in c.lower() and c != "price"]
df = df.drop(columns=leak_cols)
print("Leakage columns removed:", leak_cols)

# 1.  rebuild room_type from one‑hots if missing
if "room_type" not in df.columns or df["room_type"].isna().all():
    rt_cols = [c for c in df.columns if c.startswith("room_type_")]
    if rt_cols:
        df["room_type"] = (df[rt_cols]
                           .idxmax(axis=1)
                           .str.replace("room_type_", "", regex=False))

# keep only room_type categories with ≥50 listings for cleaner plots
rt_valid = df["room_type"].value_counts()[lambda s: s>=50].index
df.loc[~df["room_type"].isin(rt_valid), "room_type"] = "Other"

# helper to save png + pdf 
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.pdf",              bbox_inches="tight")
    plt.show()


#  FIGURE 1  –  Price distribution (log)
fig, ax = plt.subplots()
sns.histplot(df["price"], bins=60, kde=True, color="steelblue", ax=ax)
ax.set_xscale("log"); ax.set_xlabel("Nightly price (£, log)")
ax.set_title("Fig 1 Price Distribution (log scale)")
save(fig, "fig1_price_hist")

#  FIGURE 2  –  Boxplot by room_type
fig, ax = plt.subplots(figsize=(8,4))
sns.boxplot(data=df, x="price", y="room_type",
            order=df["room_type"].value_counts().index,
            showfliers=False, ax=ax)
ax.set_xscale("log"); ax.set_xlabel("Price (£, log)")
ax.set_title("Fig 2 Price by Room Type")
save(fig, "fig2_box_roomtype")

#  FIGURE 3  –  Median availability by month  (robust vs many zeros)
df["month"] = pd.to_datetime(df["last_scraped"]).dt.month_name()
months = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
med = df.groupby("month")["availability_365"].median().reindex(months)

fig, ax = plt.subplots()
sns.barplot(x=med.index, y=med.values, color="steelblue", ax=ax)
ax.set_ylabel("Median availability (days)"); ax.set_xlabel("")
ax.set_title("Fig 3 Median Availability by Month")
ax.tick_params(axis='x', rotation=45)
save(fig, "fig3_availability_month")

#  FIGURE 4  –  Hex‑bin map of average price
gdf = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326").to_crs(epsg=3857)
fig, ax = plt.subplots()
hb = ax.hexbin(gdf.geometry.x, gdf.geometry.y, C=df["price"],
               gridsize=80, reduce_C_function=np.mean,
               cmap="plasma", mincnt=4)
cx.add_basemap(ax, crs=gdf.crs); ax.set_axis_off()
fig.colorbar(hb, ax=ax).set_label("£ mean price")
ax.set_title("Fig 4 Hex‑bin Map of Average Price")
save(fig, "fig4_hex_price")

#  FIGURE 5  –  Spatial KDE (clipped, lighter)
gdf_ll = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326").cx[-0.55:0.3, 51.25:51.70]

fig, ax = plt.subplots(figsize=(6,6))
sns.kdeplot(x=gdf_ll.geometry.x, y=gdf_ll.geometry.y,
            fill=True, thresh=.05, levels=60,
            cmap="viridis", alpha=.7, bw_adjust=.4, ax=ax)
cx.add_basemap(ax, crs="EPSG:4326",
               source=cx.providers.CartoDB.Positron)
ax.set_axis_off(); ax.set_title("Fig 5 Spatial KDE – Listing Density")
save(fig, "fig5_kde_density")

#  FIGURE 6  –  Scatter accommodates vs price  (hue room_type)
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="accommodates", y="price",
                hue="room_type", alpha=.25, ax=ax)
ax.set_yscale("log"); ax.set_title("Fig 6 Capacity vs Price by Room Type")
save(fig, "fig6_scatter_accom")

#  FIGURE 7  –  Pareto of host portfolios
host_cnt = df["host_id"].value_counts(); top20 = host_cnt.head(20)
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=top20.values, y=[f"H{h}" for h in top20.index],
            color="steelblue", ax=ax)
cum = top20.cumsum()/host_cnt.sum()*100
ax2 = ax.twiny(); ax2.plot(cum.values, [f"H{h}" for h in top20.index],
                           c="crimson", marker="o")
ax.set_xlabel("Listings"); ax2.set_xlabel("Cumulative %")
ax.set_title("Fig 7 Top‑20 Host Market Share")
save(fig, "fig7_pareto_hosts")

#  FIGURE 8  –  Correlation matrix (selected numerics)
num_cols = ["price","accommodates","bathrooms","review_scores_rating",
            "availability_365","number_of_reviews"]
fig, ax = plt.subplots()
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Fig 8 Correlation Matrix (Selected Numerics)")
save(fig, "fig8_corr_matrix")

#  FIGURE 9  –  Review activity (barplot if sparse else heat‑map)
df["review_date"] = pd.to_datetime(df["last_review"], errors="coerce")
if df["review_date"].notna().sum() < 100:
    cnt = df["review_date"].dt.month_name().value_counts().reindex(months, fill_value=0)
    fig, ax = plt.subplots()
    sns.barplot(x=cnt.index, y=cnt.values, color="steelblue", ax=ax)
    ax.set_ylabel("Review count"); ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)
    ax.set_title("Fig 9 Monthly Review Counts")
else:
    cal = (df.dropna(subset=["review_date"])
             .groupby([df.review_date.dt.month_name(),
                       df.review_date.dt.day_of_week])
             .size().unstack(fill_value=0)
             .reindex(index=months,
                      columns=["Monday","Tuesday","Wednesday","Thursday",
                               "Friday","Saturday","Sunday"]))
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(cal, cmap="YlOrRd", ax=ax)
    ax.set_title("Fig 9 Review Count Heat‑map (Month × Day‑of‑Week)")
save(fig, "fig9_reviews")

#  FIGURE 10  –  Amenities word‑cloud  (skip if empty)
amenities = (df["amenities"].dropna()
               .str.lower().str.replace(r"[{}\"]","", regex=True))
if amenities.str.len().sum() > 0:
    text = " ".join([" ".join(a.split(",")) for a in amenities])
    wc = WordCloud(width=900, height=400, background_color="white",
                   collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Fig 10 Amenities Word‑Cloud")
    save(fig, "fig10_wordcloud")
else:
    print("No amenities text found – skipping Fig 10.")

print("✅  Ten figures saved (PNG & PDF).")
