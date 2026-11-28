"""
Step 0: Data Cleaning (missing values, duplicates, outliers, datetime handling)
Task 1: Average temperature for each month
Task 2: Region with the most precipitation
Task 3: Temperature trends for selected region (Almaty)
Task 4: Correlation analysis between temperature, wind speed, and precipitation
Task 5: Day with the most stable temperature for each region
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def step0(csv_path="weather.csv"):
    """
    Load weather.csv
    Display head() and info()
    Check and clean missing values
    Check and remove duplicate rows
    Check for outliers using describe() and boxplots
    Convert date to datetime and extract month features
    """
    print("=" * 80)
    print("STEP 0: SETUP & DATA CLEANING")
    print("=" * 80)

    # Load dataset
    df = pd.read_csv(csv_path)

    print("\n1. Initial Inspection:")
    print("Dataset dimensions (rows, cols):", df.shape)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nColumn Information:")
    df.info()

    # Check for missing values
    print("\n2. Missing Value Analysis:")
    missing_counts = df.isnull().sum()
    print("Missing values per column before cleaning:")
    print(missing_counts)
    if missing_counts.sum() > 0:
        df = df.dropna()
        print(f"Dropped {missing_counts.sum()} rows with missing data.")
    else:
        print("No missing values detected.")

    # Check for duplicate rows
    print("\n3. Duplicate Rows Analysis:")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows.")
    else:
        print("No duplicate records found.")

    # Convert date to datetime and extract month columns
    print("\n4. Datetime Feature Engineering:")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    print("Successfully converted 'date' to datetime format.")
    print("Extracted 'month' (integer 1-12) and 'month_name' (e.g. Jan, Feb).")
    print(df[["date", "region", "month", "month_name"]].head())

    # Outlier inspection via describe and boxplots
    print("\n5. Outlier Detection:")
    print("Descriptive statistics for numerical columns:")
    print(df[["temperature_c", "precipitation_mm", "wind_speed_kmh"]].describe())

    # Create boxplots to visually inspect distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(y=df["temperature_c"], ax=axes[0], color="skyblue")
    axes[0].set_title("Boxplot: Temperature (°C)", fontweight="bold")
    axes[0].set_ylabel("Temperature (°C)")

    sns.boxplot(y=df["precipitation_mm"], ax=axes[1], color="lightgreen")
    axes[1].set_title("Boxplot: Precipitation (mm)", fontweight="bold")
    axes[1].set_ylabel("Precipitation (mm)")

    sns.boxplot(y=df["wind_speed_kmh"], ax=axes[2], color="salmon")
    axes[2].set_title("Boxplot: Wind Speed (km/h)", fontweight="bold")
    axes[2].set_ylabel("Wind Speed (km/h)")

    plt.tight_layout()
    boxplot_file = os.path.join(OUTPUT_DIR, "step0_outlier_boxplots.png")
    plt.savefig(boxplot_file, dpi=300)
    plt.close()
    print(f"Saved outlier boxplots to '{boxplot_file}'.")
    print("Observation: Values fall within meteorologically realistic boundaries for Kazakhstan.")

    return df


def task1(df):
    """
    Group by month and calculate average temperature
    Print table and plot bar/line chart
    """
    print("\n" + "=" * 80)
    print("TASK 1: AVERAGE TEMPERATURE FOR EACH MONTH")
    print("=" * 80)

    # Group by month and compute mean temperature
    monthly_temp = df.groupby(["month", "month_name"])["temperature_c"].mean().reset_index()
    monthly_temp = monthly_temp.sort_values(by="month")

    print("\nMonthly Average Temperatures:")
    for _, row in monthly_temp.iterrows():
        print(f"Month {int(row['month']):>2} ({row['month_name']:<3}): {row['temperature_c']:>6.2f} °C")

    # Plot monthly temperature
    plt.figure(figsize=(10, 5))
    plt.bar(
        monthly_temp["month_name"],
        monthly_temp["temperature_c"],
        color="cornflowerblue",
        edgecolor="black",
        alpha=0.85,
        label="Monthly Mean Bar"
    )
    plt.plot(
        monthly_temp["month_name"],
        monthly_temp["temperature_c"],
        color="crimson",
        marker="o",
        linewidth=2.5,
        label="Seasonal Temperature Curve"
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.2, label="Freezing Point (0°C)")
    plt.title("Average Monthly Temperature Across All Regions (2024)", fontsize=14, fontweight="bold")
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Temperature (°C)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "task1_monthly_average_temperature.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved Task 1 chart to '{chart_path}'.")

    return monthly_temp


def task2(df):
    """
    Group by region and sum precipitation
    Sort from highest to lowest and print
    Bar chart comparing total precipitation across regions
    """
    print("\n" + "=" * 80)
    print("TASK 2: REGION WITH THE MOST PRECIPITATION")
    print("=" * 80)

    # Group by region and calculate total precipitation
    region_precip = df.groupby("region")["precipitation_mm"].sum().reset_index()
    region_precip.columns = ["region", "total_precipitation_mm"]
    region_precip_sorted = region_precip.sort_values(by="total_precipitation_mm", ascending=False)

    print("\nTotal Precipitation by Region (Ranked):")
    for rank, (_, row) in enumerate(region_precip_sorted.iterrows(), start=1):
        print(f"{rank}. {row['region']:<12}: {row['total_precipitation_mm']:>7.2f} mm")

    top_region = region_precip_sorted.iloc[0]["region"]
    top_precip = region_precip_sorted.iloc[0]["total_precipitation_mm"]
    print(f"\nRESULT: The region with the most precipitation is '{top_region}' with {top_precip:.2f} mm.")

    # Create visualization
    plt.figure(figsize=(9, 5))
    bar_colors = ["#2b5c8f", "#4682b4", "#5c9edc", "#87ceeb", "#b0e0e6"]
    bars = plt.bar(
        region_precip_sorted["region"],
        region_precip_sorted["total_precipitation_mm"],
        color=bar_colors[:len(region_precip_sorted)],
        edgecolor="black"
    )

    # Add numeric labels on bars
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 5, f"{h:.1f} mm", ha="center", va="bottom", fontweight="bold")

    plt.title("Total Precipitation by Region (Full Period)", fontsize=14, fontweight="bold")
    plt.xlabel("Region", fontsize=12)
    plt.ylabel("Total Precipitation (mm)", fontsize=12)
    plt.ylim(0, region_precip_sorted["total_precipitation_mm"].max() * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "task2_total_precipitation_by_region.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved Task 2 chart to '{chart_path}'.")

    return region_precip_sorted


def task3(df, region_name="Almaty"):
    """
    Filter dataset for specific region
    Line chart of temperature_c over date with title, labels, and grid
    """
    print("\n" + "=" * 80)
    print(f"TASK 3: TEMPERATURE TRENDS FOR SELECTED REGION ({region_name.upper()})")
    print("=" * 80)

    # Filter for the region and sort by date
    reg_df = df[df["region"] == region_name].sort_values(by="date").copy()

    print(f"\nSummary for {region_name}:")
    print(f"- Total days recorded: {len(reg_df)}")
    print(f"- Minimum Temperature: {reg_df['temperature_c'].min():.1f} °C")
    print(f"- Maximum Temperature: {reg_df['temperature_c'].max():.1f} °C")
    print(f"- Mean Temperature:    {reg_df['temperature_c'].mean():.1f} °C")

    # Calculate 7-day rolling average for smooth trendline
    reg_df["temp_7day_rolling"] = reg_df["temperature_c"].rolling(window=7, center=True).mean()

    # Plot
    plt.figure(figsize=(13, 5))
    plt.plot(
        reg_df["date"],
        reg_df["temperature_c"],
        color="#d95f02",
        linewidth=1.2,
        alpha=0.85,
        label="Daily Temperature"
    )
    plt.plot(
        reg_df["date"],
        reg_df["temp_7day_rolling"],
        color="navy",
        linewidth=2.0,
        label="7-Day Moving Average"
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Freezing Point (0°C)")

    plt.title(f"Daily Temperature Trend in {region_name} (2024)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f"task3_temp_trends_{region_name.lower()}.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved Task 3 chart to '{chart_path}'.")

    return reg_df


def task4(df):
    """
    Select numeric columns: temperature, precipitation, wind_speed
    Calculate and print correlation matrix
    Seaborn heatmap with annot=True
    """
    print("\n" + "=" * 80)
    print("TASK 4: CORRELATION ANALYSIS")
    print("=" * 80)

    numeric_cols = ["temperature_c", "precipitation_mm", "wind_speed_kmh"]
    corr_matrix = df[numeric_cols].corr()

    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))

    # Heatmap visualization
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix: Temperature, Wind Speed & Precipitation", fontsize=12, fontweight="bold")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "task4_correlation_heatmap.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved Task 4 heatmap to '{chart_path}'.")

    return corr_matrix


def task5(df):
    """
    Defined as the date with the minimal day-to-day absolute temperature difference:
    """
    print("\n" + "=" * 80)
    print("TASK 5: DAY WITH THE MOST STABLE TEMPERATURE IN EACH REGION")
    print("=" * 80)

    # Sort sequentially
    df_sorted = df.sort_values(by=["region", "date"]).copy()
    df_sorted["prev_day_temp"] = df_sorted.groupby("region")["temperature_c"].shift(1)
    df_sorted["daily_temp_diff"] = (df_sorted["temperature_c"] - df_sorted["prev_day_temp"]).abs()

    # Drop the first day for each region where previous day is NaN
    df_valid = df_sorted.dropna(subset=["daily_temp_diff"])

    unique_regions = df_valid["region"].unique()
    stable_days = []

    for reg in unique_regions:
        reg_data = df_valid[df_valid["region"] == reg]
        min_idx = reg_data["daily_temp_diff"].idxmin()
        best_row = reg_data.loc[min_idx]

        stable_days.append({
            "Region": reg,
            "Date": best_row["date"].strftime("%Y-%m-%d"),
            "Today_Temp_C": best_row["temperature_c"],
            "Yesterday_Temp_C": best_row["prev_day_temp"],
            "Abs_Difference_C": round(best_row["daily_temp_diff"], 3)
        })

    stable_df = pd.DataFrame(stable_days)

    print("\nTable of Most Stable Temperature Days per Region:")
    print(stable_df.to_string(index=False))

    return stable_df


def main():
    print("Starting End-Term Weather Analysis Project Execution...\n")

    df = step0("weather.csv")
    task1(df)
    task2(df)
    task3(df, region_name="Almaty")
    task4(df)
    task5(df)

    print("\n" + "=" * 80)
    print("All tasks completed successfully! All charts saved in 'figures/' directory.")
    print("=" * 80)


if __name__ == "__main__":
    main()
