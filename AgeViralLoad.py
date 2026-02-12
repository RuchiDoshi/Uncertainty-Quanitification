"""
    Goal is to examine whether viral load diffes by age group and how that changes over time
    To analyze whether cetain age groups consistently show higher viral intensity and whether that
        shift during waves
    Line plot:
        X-axis: time/date
        Y-axis: average CtValue (axis decreasing since higher Ct is weaker viral load)
        Multiple lines: one per age range
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("1.1_Final_Dataset_Patient_Level.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    
    #Group by Age - ten year bins
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df.dropna(subset=["Age"])

    df["Age_Group"] = (df["Age"] // 10) * 10
    df["Age_Group_Label"] = df["Age_Group"].astype(str) + "-" + (df["Age_Group"] + 9).astype(str)

    # get the average CtValue over a month per age group 
    df["Ct_Value"] = pd.to_numeric(df["Ct_Value"], errors="coerce")


    monthly_ct = (
    df.groupby(["Month", "Age_Group_Label"], as_index=False)
      .agg(Avg_Ct=("Ct_Value", "mean"))
    )

    plt.figure(figsize=(12,6))

    for group in monthly_ct["Age_Group_Label"].unique():
        subset = monthly_ct[monthly_ct["Age_Group_Label"] == group]
        plt.plot(subset["Month"], subset["Avg_Ct"], label=group)

    plt.xlabel("Month")
    plt.ylabel("Average Ct Value")
    plt.title("Monthly Average Ct Value by Age Group")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()