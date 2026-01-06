import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import lstsq
from sklearn.model_selection import train_test_split
import cvxpy as cp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# %%
df_data = pd.read_parquet("training_data/initial_dataset.parquet")
df_train_detached = pd.read_parquet(
    "training_data/data_detached_with_weather.parquet")
type(df_data)
type(df_train_detached)
df_train_detached

unique_ids = df_train_detached["Property_ID"].unique()
id_use = unique_ids[0]
df_house = pd.read_csv('training_data/home_characteristics.csv')

# List of columns to check for missingness
cols = [
    "Bedrooms", "Floor_Height", "Habitable_Rooms", "House_Age",
    "House_Form", "No_Storeys", "No_Underfloor",
    "Total_Floor_Area", "Wall_Type", "MCS_DHWAnnual", "HP_Size_kW"
]

# calculate min capacity in home
cp_air = 1.005  # kJ/kgK
rho_air = 1.225  # kg/m^3
kj_to_kWh = 1 / 3600
df_house["Volume"] = df_house["Total_Floor_Area"] * df_house["Floor_Height"]
df_house["Min Capacity"] = df_house["Volume"] * cp_air * rho_air * kj_to_kWh

df_home_values = df_house[df_house["Property_ID"].isin(unique_ids)]

summary = df_home_values[["Property_ID", "Min Capacity"]].drop_duplicates()
print(summary)

home_dict = df_home_values.set_index("Property_ID").to_dict(orient="index")
# %%
df_single = df_train_detached[
    df_train_detached["Property_ID"] == id_use].copy()

# re-adjust Heat Pump Diff and add temp differences
df_single["Heat_Pump_Energy_Output_Diff"] = df_single[
    "Heat_Pump_Energy_Output"].diff()
df_single["Internal_Temperature_Diff"] = df_single[
    "Internal_Air_Temperature"].diff()
df_single["Internal_Ambient_Temperature_Diff"] = \
    (df_single["External_Air_Temperature"] -
     df_single["Internal_Air_Temperature"])

# 1. Drop columns with almost all missing data (e.g., more than 90% missing)
threshold = 0.90 * len(df_single)
df_single_cleaned = df_single.dropna(axis=1, thresh=threshold)
# print("Columns dropped due to high missing values:")
# print(df_single.columns.difference(df_single_cleaned.columns).tolist())

df_single = df_single_cleaned

# print("\nColumns remaining after dropping highly missing columns:")
# print(df_single.columns.tolist())

df_single = df_single.set_index('Timestamp')
df_single = df_single.sort_index()

# Apply interpolation with a limit of 4 (for 2 hours of half-hourly data)
numeric_cols = df_single.select_dtypes(include=['number']).columns
df_single_numeric_interpolated = df_single[numeric_cols].interpolate(
    method='time', limit=4, limit_direction='both')

df_single_interpolated = df_single.copy()
df_single_interpolated[numeric_cols] = df_single_numeric_interpolated

# After interpolation, drop rows that still contain NaN values (meaning they were missing for > 2 hours)
initial_rows = len(df_single_interpolated)
df_single_processed = df_single_interpolated.dropna()
# %%
df_heating_single = df_single_processed.copy()

# Time range
t_start = pd.to_datetime("2021-10-01 00:00:00")
t_end = pd.to_datetime("2022-10-31 23:59:00")
df_heating_only = df_heating_single[
    (df_heating_single.index >= t_start) & (df_heating_single.index <= t_end)
    ].copy()

df_heating_only.drop("Property_ID", axis=1, inplace=True)
df_heating_only.drop("half-hour", axis=1, inplace=True)
df_heating_only.drop("Date", axis=1, inplace=True)
df_heating_only.drop("has_data", axis=1, inplace=True)

t_step_try = [1, 3, 6, 12, 24]

figures = []  # collect handles


for i in t_step_try:
    df_numeric   = df_heating_only.select_dtypes(include='number')
    df_resampled = df_numeric.resample(f'{i}h').mean()

    ΔTa = df_resampled["Internal_Ambient_Temperature_Diff"]
    Q   = df_resampled["Heat_Pump_Energy_Output_Diff"]
    ΔTi = df_resampled["Internal_Temperature_Diff"]
    Tamb= df_resampled["External_Air_Temperature"]

    # 1) Ambient‐delta vs Q
    fig1 = plt.figure()
    plt.scatter(ΔTa, Q)
    plt.title(f"Ambient Delta vs Q (Δt={i}h)")
    plt.xlabel("Δ Temperature")
    plt.ylabel("Heat Input Q")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    figures.append(fig1)

    # 2) Internal‐delta vs Q
    fig2 = plt.figure()
    plt.scatter(ΔTi, Q)
    plt.title(f"Internal Delta vs Q (Δt={i}h)")
    plt.xlabel("Δ Temperature")
    plt.ylabel("Heat Input Q")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    figures.append(fig2)

    # 3) 3D scatter
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    p  = ax.scatter(ΔTi, Q, ΔTa, c=Tamb, cmap='viridis', s=40, depthshade=True)
    ax.set_xlabel("Δ Temperature")
    ax.set_ylabel("Heat Input Q")
    ax.set_zlabel("Ambient Temperature (°C)")
    fig3.colorbar(p, ax=ax, pad=0.1, label='Ambient T')
    plt.title(f"3D Scatter (Δt={i}h)")
    plt.tight_layout()
    plt.show(block=False)
    figures.append(fig3)

# Now block here until you manually close all figures:
plt.show()