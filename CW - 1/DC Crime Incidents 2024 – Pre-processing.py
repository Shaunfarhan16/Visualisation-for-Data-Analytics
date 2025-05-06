import pandas as pd

# ── load ────────────────────────────────────────────────────────────────────────
dc_path = r"dc_crime_2024.csv"                    # ← update if needed
dc = pd.read_csv(dc_path)
dc.columns = dc.columns.str.upper()               # normalise headers

# ── 1. parse timestamps & keep calendar-year 2024 ──────────────────────────────
date_col = "REPORT_DAT"
dc[date_col] = pd.to_datetime(dc[date_col], errors="coerce")
dc = dc[dc[date_col].dt.year == 2024]

# ── 2. drop rows without valid GPS  & enforce bounding-box ─────────────────────
lat, lon = "LATITUDE", "LONGITUDE"
dc = dc.dropna(subset=[lat, lon])
dc = dc[dc[lat].between(38.80, 39.00) & dc[lon].between(-77.12, -76.90)]

# ── 3. derive temporal features for heat-maps etc. ─────────────────────────────
dc["HOUR"]    = dc[date_col].dt.hour
dc["WEEKDAY"] = dc[date_col].dt.day_name()

# ── 4. consolidate rare offences (<1 % frequency) ──────────────────────────────
if "OFFENSE" in dc.columns:
    freq = dc["OFFENSE"].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    dc["OFFENSE"] = dc["OFFENSE"].where(~dc["OFFENSE"].isin(rare), other="Other")

# ── 5. drop low-utility columns (if present) ───────────────────────────────────
dc = dc.drop(columns=[c for c in ("METHOD", "NEIGHBORHOOD_CLUSTER") if c in dc.columns])

print("Clean DC-crime dataframe shape:", dc.shape)
