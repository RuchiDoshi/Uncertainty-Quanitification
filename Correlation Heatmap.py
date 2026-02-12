import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import unicodedata



city_path = "/content/2.1_2.2_Final_Dataset_City_Level (1).xlsx"
patient_path = "/content/1.1_Final_Dataset_Patient_Level.xlsx"

city_df = pd.read_excel(city_path, sheet_name="2.1_City_Daily_Missing")
patient_df = pd.read_excel(patient_path)



def clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).strip().upper()
    x = unicodedata.normalize('NFKD', x)\
        .encode('ascii', errors='ignore')\
        .decode('utf-8')
    return x

patient_df["City"] = patient_df["City"].apply(clean_text)
city_df["City"] = city_df["City"].apply(clean_text)

patient_df["State"] = patient_df["State"].apply(clean_text)
city_df["State"] = city_df["State"].apply(clean_text)



patient_df["Date"] = pd.to_datetime(patient_df["Date"], errors="coerce").dt.normalize()
city_df["Date"] = pd.to_datetime(city_df["Date"], errors="coerce").dt.normalize()


patient_df = patient_df[
    (patient_df["Ct_Value"] >= 10) &
    (patient_df["Ct_Value"] <= 40)
]

patient_df["Male"] = patient_df["Sex"].map({"M": 1, "F": 0})



agg_patient = (
    patient_df.groupby(["City", "State", "Date"])
    .agg(
        Mean_CT=("Ct_Value", "mean"),
        Median_CT=("Ct_Value", "median"),
        CT_Variance=("Ct_Value", "var"),
        Low_CT_Prop=("Ct_Value", lambda x: np.mean(x < 20)),
        Very_Low_CT_Prop=("Ct_Value", lambda x: np.mean(x < 15)),
        Mean_Age=("Age", "mean"),
        Male_Prop=("Male", "mean"),
        Patient_Count=("PCR_Id", "count"),
    )
    .reset_index()
)

print("Shape of agg_patient:", agg_patient.shape)
print("Shape of city_df:", city_df.shape)


# merge

merged_df = pd.merge(
    agg_patient,
    city_df,
    on=["City", "State", "Date"],
    how="inner"
)

print("Shape after merge:", merged_df.shape)

if merged_df.empty:
    raise ValueError("Merge failed — no matching City/State/Date keys found.")


# variable

variables = [
    "Mean_CT",
    "CT_Variance",
    "Low_CT_Prop",
    "NewCases",
    "NewDeaths",
    "TotalCases_100k_inhab",
    "Stringency_Index"
]

# log
for col in ["NewCases", "NewDeaths"]:
    merged_df[col] = np.log1p(merged_df[col])


# correlation

corr_matrix = merged_df[variables].corr(method="spearman")


# plot

plt.figure(figsize=(12, 8))

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

plt.title("Cross-Level Correlation Heatmap (City-Date Aggregated)", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
