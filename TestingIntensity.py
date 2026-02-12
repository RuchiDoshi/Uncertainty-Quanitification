"""
    Goal is to determine whether higher case counts are driven by increased transmission or simply increased testing.
    Will create a scatter plot:
        X-axis: Number of PCR tests per 100k per week
        Y-axis: New cases per 100k per week
        Each dot = (city during a certain week)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("2.1_City_Daily_Missing.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # 1) Daily tests from cumulative Patience_Count
    df["DailyTests"] = df.groupby("City")["Patience_Count"].diff()

    # 2) Clean negative test amounts
    df.loc[df["DailyTests"] < 0, "DailyTests"] = np.nan

    # 3) Estimate population using data
    df.loc[df["TotalCases_100k_inhab"] <= 0, "TotalCases_100k_inhab"] = np.nan
    df["Population"] = df["TotalCases"] / df["TotalCases_100k_inhab"]

    # 4) Weekly aggregation
    df["Week"] = df["Date"].dt.to_period("W").dt.start_time

    weekly = (
        df.groupby(["City", "Week"], as_index=False)
          .agg(
              WeeklyTests=("DailyTests", "sum"),
              WeeklyCases=("NewCases", "sum"),
              Population=("Population", "median"),
              Stringency=("Stringency_Index", "mean"),
          )
    )

    # 5) Per 100k rates
    weekly["Tests_per_100k"] = weekly["WeeklyTests"] / weekly["Population"] * 100000
    weekly["Cases_per_100k"] = weekly["WeeklyCases"] / weekly["Population"] * 100000

    # 6) Clean rows
    weekly = weekly.replace([np.inf, -np.inf], np.nan)
    weekly = weekly.dropna(subset=["Tests_per_100k", "Cases_per_100k"])
    weekly = weekly[weekly["WeeklyTests"] >= 10]  # optional: remove tiny-test weeks

    # 7) Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(weekly["Tests_per_100k"], weekly["Cases_per_100k"], alpha=0.3, s=10)
    plt.xlabel("PCR Tests per 100k (weekly)")
    plt.ylabel("New Cases per 100k (weekly)")
    plt.title("Testing intensity vs observed case rate (city-week)")
    plt.grid(True)
    plt.show()

    plt.xscale("log")
    plt.yscale("log")

if __name__ == "__main__":
    main()
