
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"C:\Users\emma\Downloads\2.1_2.2_Final_Dataset_City_Level (1).xlsx"
sheet_name = "2.1_City_Daily_Missing"
city_df = pd.read_excel(file_path, sheet_name=sheet_name)
file_path1 = r"C:\Users\emma\Downloads\1.1_Final_Dataset_Patient_Level.xlsx"
patient_df = pd.read_excel(file_path1)

#  Date columns are datetime
patient_df["Date"] = pd.to_datetime(patient_df["Date"])
city_df["Date"] = pd.to_datetime(city_df["Date"])


# Remove invalid CT values (<10 or >40)
patient_df = patient_df[
    (patient_df["Ct_Value"] >= 10) &
    (patient_df["Ct_Value"] <= 40)
]

# Male = 1, Female = 0
patient_df["Male"] = patient_df["Sex"].map({"M": 1, "F": 0})


#aggregate patient level  to city date
agg_patient = patient_df.groupby(
    ["City", "State", "Date"]
).agg(
    Mean_CT=("Ct_Value", "mean"),
    Median_CT=("Ct_Value", "median"),
    CT_Variance=("Ct_Value", "var"),
    Low_CT_Prop=("Ct_Value", lambda x: np.mean(x < 20)),
    Very_Low_CT_Prop=("Ct_Value", lambda x: np.mean(x < 15)),
    Mean_Age=("Age", "mean"),
    Male_Prop=("Male", "mean"),
    Patient_Count=("PCR_Id", "count")
).reset_index()


#merge two sets
merged_df = pd.merge(
    agg_patient,
    city_df,
    on=["City", "State", "Date"],
    how="inner"
)


#variables
variables = [
    # Patient
    "Mean_CT",
    "CT_Variance",
    "Low_CT_Prop",
    "Mean_Age",
    "Male_Prop",
    "NewCases",
    "TotalCases_100k_inhab",
    "NewDeaths",
    "Hosp_Count",
    "Hosp_Death_Rate",
    "Vax_AllDoses",
    "Stringency_Index",
    "TotalDeaths_by_TotalCases",
    "Aver_Hosp_Stay"
]

# Remove rows with missing values
#corr_df = merged_df[variables].dropna()


#correlation
corr_matrix = corr_df.corr(method="spearman")

#heatMap
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Cross-Level Correlation Heatmap (City-Date Aggregated)", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()