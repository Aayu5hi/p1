import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


#plt.ion()

# CSV dataset load
#df = pd.read_csv("traffic_dataset.csv")
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "traffic_dataset.csv")

df = pd.read_csv(file_path)

# basic data info x4
print("Dataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print(df.describe())


# ---------- VEHICLE TYPE SEPARATION ----------

public_vehicles = ["ambulance", "fire", "patrol"]
private_vehicles = ["car", "cars", "bike", "bikes", "van", "vans", "truck", "trucks"]

def extract_vehicle_count(vehicle_str, vehicle_list):
    if pd.isna(vehicle_str):
        return 0

    total = 0
    parts = vehicle_str.split(",")

    for part in parts:
        match = re.search(r'(\d+)\s+([\w]+)', part.strip())
        if match:
            count = int(match.group(1))
            vehicle = match.group(2).lower()

            if vehicle in vehicle_list:
                total += count

    return total

df["Public_Vehicle_Count"] = df["Vehicle Types Detected"].apply(
    lambda x: extract_vehicle_count(x, public_vehicles)
)

df["Private_Vehicle_Count"] = df["Vehicle Types Detected"].apply(
    lambda x: extract_vehicle_count(x, private_vehicles)
)
# ---------- VEHICLE TYPE SEPARATION ----------

print(df[["Vehicle Types Detected", "Public_Vehicle_Count", "Private_Vehicle_Count"]].head())



# minute-wise time index fix
df["minute_index"] = range(len(df))

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Public_Vehicle_Count",
    y="Avg Speed (km/h)",
    alpha=0.6
)
plt.title("Effect of Public Vehicle Density on Traffic Speed")
plt.xlabel("Public Vehicle Count")
plt.ylabel("Average Speed (km/h)")
plt.show()


plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Congestion Level",
    y="Public_Vehicle_Count"
)
plt.title("Public Vehicle Density Across Congestion Levels")
plt.xlabel("Congestion Level")
plt.ylabel("Public Vehicle Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Private_Vehicle_Count",
    y="Avg Speed (km/h)",
    label="Private",
    alpha=0.5
)
sns.scatterplot(
    data=df,
    x="Public_Vehicle_Count",
    y="Avg Speed (km/h)",
    label="Public",
    alpha=0.5
)
plt.title("Public vs Private Vehicle Impact on Speed")
plt.xlabel("Vehicle Count")
plt.ylabel("Average Speed (km/h)")
plt.legend()
plt.show()


#figure 1
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Vehicle Density (%)",
    y="Avg Speed (km/h)",
    hue="Congestion Level",
    alpha=0.6
)

plt.title("Effect of Vehicle Density on Traffic Speed")
plt.xlabel("Vehicle Density (%)")
plt.ylabel("Average Speed (km/h)")
plt.show()
#plt.close()



# figure-2 Rolling average of speed (30 minutes)
df["speed_rolling"] = df["Avg Speed (km/h)"].rolling(window=30).mean()

plt.figure(figsize=(10, 5))
plt.plot(df["minute_index"], df["speed_rolling"])
plt.xlabel("Time (minutes since start)")
plt.ylabel("Average Speed (km/h)")
plt.title("Traffic Speed Trend Over Time (30-Minute Rolling Average)")
plt.show()



# figure-4
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Congestion Level",
    y="Avg Speed (km/h)"
)
plt.title("Speed Across Different Congestion Levels")
plt.show()

# figure-3
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Congestion Level")
plt.title("Distribution of Traffic Congestion Levels")
plt.show()


"""
# 4. Convert Timestamp column to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# 5. Extract hour from timestamp (for time-based analysis)
df["hour"] = df["Timestamp"].dt.hour

# 6. Check vehicle types and their counts
print("\nVehicle type distribution:")
print(df["Vehicle type"].value_counts())
"""