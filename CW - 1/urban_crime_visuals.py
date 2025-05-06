# -*- coding: utf-8 -*-
# ------------------------------------------------------
#  DC Crime 2024 – Three Key Visualisations in one script
# ------------------------------------------------------
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
plt.rcParams['figure.dpi'] = 110

CSV_PATH = r"C:\\Visualization of Data Analytics\\columbia_crime_2024.csv"          # <-- update to your file

# ---------- 1.  load & preprocess --------------------------------------------
dc = pd.read_csv(CSV_PATH)
dc.columns = dc.columns.str.upper()
dc['REPORT_DAT'] = pd.to_datetime(dc['REPORT_DAT'], errors='coerce')
dc = dc[(dc['REPORT_DAT'].dt.year == 2024) &
        dc['LATITUDE'].between(38.80, 39.00) &
        dc['LONGITUDE'].between(-77.12, -76.90)]

# derive temporal helpers
dc['HOUR']    = dc['REPORT_DAT'].dt.hour
dc['WEEKDAY'] = dc['REPORT_DAT'].dt.day_name()
dc['MONTH']   = dc['REPORT_DAT'].dt.strftime('%b')

# ---------- 2.  Fig A  –  Monthly box-plot -----------------------------------
daily = (dc.set_index('REPORT_DAT')
           .resample('D').size()
           .reset_index(name='COUNT'))
daily['MONTH'] = daily['REPORT_DAT'].dt.strftime('%b')

plt.figure(figsize=(10,4))
sns.boxplot(data=daily, x='MONTH', y='COUNT',
            order=['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec'])
plt.title('Fig A  Monthly Distribution of Daily Incidents – DC 2024')
plt.ylabel('Incidents per day')
plt.tight_layout()
plt.savefig('FigA_Box.png')

# ---------- 3.  Fig B  –  Hour × Weekday heat-map ----------------------------
pivot = (dc.pivot_table(index='WEEKDAY', columns='HOUR',
                        values='X', aggfunc='count')
           .reindex(['Monday','Tuesday','Wednesday','Thursday',
                     'Friday','Saturday','Sunday']))

plt.figure(figsize=(9,4))
sns.heatmap(pivot, cmap='YlOrRd')
plt.title('Fig B  Heat-map of Incidents by Hour & Weekday (2024)')
plt.xlabel('Hour of day'); plt.ylabel('Weekday')
plt.tight_layout()
plt.savefig('FigB_Heat.png')

# ---------- 4.  Fig C  –  Spatial KDE hotspot map ----------------------------
plt.figure(figsize=(6,6))
sns.kdeplot(x=dc['LONGITUDE'], y=dc['LATITUDE'],
            shade=True, bw_adjust=.4, cmap='Reds', thresh=.05)
plt.scatter(dc['LONGITUDE'], dc['LATITUDE'],
            s=1, alpha=.04, color='k')          # context dots
plt.title('Fig C  Spatial KDE Hotspots – All Offences 2024')
plt.gca().set_aspect('equal'); plt.axis('off')
plt.tight_layout()
plt.savefig('FigC_KDE.png')
plt.show()

plt.figure(figsize=(7,3))
counts = dc['OFFENSE'].value_counts().head(10)
sns.barplot(x=counts.values, y=counts.index, palette='viridis')
plt.title('Top-10 Offence Categories – DC 2024')
plt.xlabel('Number of Incidents'); plt.tight_layout()


hourly = dc.groupby('HOUR').size()
plt.figure(figsize=(8,3))
hourly.plot(marker='o'); plt.xticks(range(0,24))
plt.title('Average Incidents by Hour of Day – 2024')
plt.ylabel('Incidents'); plt.tight_layout()

