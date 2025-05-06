import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Cleveland Heart Disease Dataset
# ---------------------------------------

# Load the Cleveland Heart Disease CSV
heart_file = 'C:\\Visualization of Data Analytics\\Heart_disease_cleveland_new.csv'  # Update path as needed
heart_df = pd.read_csv(heart_file)

# Inspect columns
print("Heart Disease columns:", heart_df.columns.tolist())

# Impute missing 'ca' and 'thal' with mode
for col in ['ca', 'thal']:
    if col in heart_df.columns:
        heart_df[col] = heart_df[col].fillna(heart_df[col].mode()[0])

# Clip outliers in continuous features at 1st/99th percentiles
cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in cont_cols:
    if col in heart_df.columns:
        lower, upper = heart_df[col].quantile([0.01, 0.99])
        heart_df[col] = heart_df[col].clip(lower=lower, upper=upper)

# Z-score normalization
scaler = StandardScaler()
heart_df[cont_cols] = scaler.fit_transform(heart_df[cont_cols])

# One-hot encode categorical features
cat_cols = ['cp', 'restecg', 'slope', 'sex', 'fbs', 'exang']
heart_df = pd.get_dummies(heart_df, columns=[c for c in cat_cols if c in heart_df.columns])

# Split into features and target
if 'target' in heart_df.columns:
    X_hd = heart_df.drop(columns=['target'])
    y_hd = heart_df['target']
else:
    raise KeyError("Column 'target' not found in heart disease dataset.")

# Train-test split (80:20, stratified)
X_hd_train, X_hd_test, y_hd_train, y_hd_test = train_test_split(
    X_hd, y_hd, test_size=0.2, stratify=y_hd, random_state=42
)

print("Heart Disease - train/test shapes:", X_hd_train.shape, X_hd_test.shape)

# ---------------------------------------
# 2. District of Columbia Crime Incidents Dataset
# ---------------------------------------

# Load the DC Crime CSV
crime_file = 'C://Visualization of Data Analytics//columbia_crime_2024.csv'  # Update path as needed
crime_df = pd.read_csv(crime_file)

# Inspect columns
print("Crime dataset columns:", crime_df.columns.tolist())

# Identify datetime column
date_cols = [c for c in crime_df.columns if 'date' in c.lower()]
if not date_cols:
    raise KeyError("No date column detected. Check your CSV headers.")
date_col = date_cols[0]
crime_df[date_col] = pd.to_datetime(crime_df[date_col], errors='coerce')

# Derive temporal features
crime_df['hour'] = crime_df[date_col].dt.hour
crime_df['day_of_week'] = crime_df[date_col].dt.day_name()
crime_df['month'] = crime_df[date_col].dt.month

# Identify latitude/longitude columns
lat_cols = [c for c in crime_df.columns if 'lat' in c.lower()]
lon_cols = [c for c in crime_df.columns if 'lon' in c.lower()]
if not lat_cols or not lon_cols:
    raise KeyError("Latitude/Longitude columns not found.")
lat_col = lat_cols[0]
lon_col = lon_cols[0]

# Drop rows with missing coordinates
crime_df = crime_df.dropna(subset=[lat_col, lon_col])

# Filter to DC bounding box
crime_df = crime_df[
    (crime_df[lat_col].between(38.80, 39.00)) &
    (crime_df[lon_col].between(-77.12, -76.90))
]

# Consolidate rare offense categories (<1% frequency)
if 'Offense' in crime_df.columns:
    freq = crime_df['Offense'].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    crime_df['Offense'] = crime_df['Offense'].apply(lambda x: 'Other' if x in rare else x)
else:
    print("Column 'Offense' not found—skipping consolidation.")

# Drop low-utility columns if present
drop_cols = ['Method', 'Neighborhood Cluster']
crime_df = crime_df.drop(columns=[c for c in drop_cols if c in crime_df.columns])

print("Crime dataset shape after preprocessing:", crime_df.shape)

# ---------------------------------------
# 3. Basic Exploratory Analysis Templates
# ---------------------------------------

# Heart Disease: Correlation matrix heatmap
plt.figure(figsize=(8, 6))
corr = heart_df.corr()
plt.imshow(corr, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Heart Disease Feature Correlation Matrix')
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.tight_layout()
plt.show()

# Crime: Crime count by offense bar chart
if 'Offense' in crime_df.columns:
    offense_counts = crime_df['Offense'].value_counts().nlargest(10)
    offense_counts.plot(kind='bar', figsize=(8, 4))
    plt.title('Top 10 Crime Types in DC (2024)')
    plt.xlabel('Offense')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Crime: Time series of daily crime counts
plt.figure(figsize=(10, 4))
daily_counts = crime_df.set_index(date_col).resample('D').size()
daily_counts.plot()
plt.title('Daily Crime Incidents in DC (2024)')
plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.tight_layout()
plt.show()

