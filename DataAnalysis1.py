import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

"""
Algorithim to find basic statistical data about dependent variables
 New cases, total cases/100 inhabitants, total deaths/total cases
"""
def primary_predictors(df, cities):
    #separate dates just by year
    df['Year'] = df['Date'].dt.year

    #group by state, city, and date and find the minimum, maximum, mean and standard devation for each year in each city
    grouped = df.groupby(['State', 'City', 'Year'])[['Ct_Value', 'NewCases']].agg(['max', 'min', 'mean', 'std'])
    
    #flatten multi level headers
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    #Reset the index so State, City, and Year become normal columns
    final_df = grouped.reset_index()

    # 3. Export to CSV
    # index=False -> don't get an extra column of row numbers
    final_df.to_csv('City_Yearly_Stats.csv', index=False)
    
    #create rolling cases alongside ct values visualization for all cities included in parameter
    for city_name in cities:
        #creates smallen dataframe with just values from desired city - sorted by date
        city_df = df[df['City'] == city_name].sort_values('Date')

        # takes 7 day averages of Ct and Case data to account for unnecessary noise that may occur due to weekends etc
        city_df['Cases_Smooth'] = city_df['NewCases'].rolling(window=7).mean()
        city_df['Ct_Smooth'] = city_df['Ct_Value'].rolling(window=7).mean()

        # 2. Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Primary Axis: New Cases
        ax1.set_xlabel('Date')
        ax1.set_ylabel('New Cases (7-day Average)', color='blue')
        ax1.plot(city_df['Date'], city_df['Cases_Smooth'], color='blue', label='Cases')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Secondary Axis: Ct Value
        ax2 = ax1.twinx()
        ax2.set_ylabel('Ct Value (Inverted: Lower = Higher Viral Load)', color='red')
        ax2.plot(city_df['Date'], city_df['Ct_Smooth'], color='red', linestyle=':', label='Ct Value')
        ax2.tick_params(axis='y', labelcolor='red')

        # CRITICAL: Invert the Ct axis
        ax2.invert_yaxis() 

        plt.title(f'Primary Predictor Analysis: Ct Value vs. New Cases in {city_name}')
        plt.show()
    
    return grouped

if __name__ == "__main__":
    df = pd.read_csv('2.2_City_Daily_NoMissing.csv')

    date_toweekday = {}

    start_date = date(2020, 1, 1)
    end_date = date(2023, 12, 31)


    print(df.shape)
    ##set data column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # 1. Replace the string 'NA' with numpy's NaN
    df['Ct_Value'] = df['Ct_Value'].replace('NA', np.nan)
    # 2. Now check the count of NaN values
    print(df['Ct_Value'].isnull().sum())

    # Convert desired variables to numeric, coercing any remaining non-numeric strings to NaN
    df['Ct_Value'] = pd.to_numeric(df['Ct_Value'], errors='coerce')
    df['NewCases'] = pd.to_numeric(df['NewCases'], errors='coerce')
    df['TotalCases_100k_inhab'] = pd.to_numeric(df['TotalCases_100k_inhab'], errors='coerce')
    df['TotalDeaths_by_TotalCases'] = pd.to_numeric(df['TotalDeaths_by_TotalCases'], errors='coerce')
    df['Hosp_Count'] = pd.to_numeric(df['Hosp_Count'], errors='coerce')
    df['Aver_Hosp_Stay'] = pd.to_numeric(df['Aver_Hosp_Stay'], errors='coerce')
    df['Stringency_Index'] = pd.to_numeric(df['Stringency_Index'], errors='coerce')
    df['Vax_Dose1'] = pd.to_numeric(df['Vax_Dose1'], errors='coerce')
    df['Vax_Dose2'] = pd.to_numeric(df['Vax_Dose2'], errors='coerce')
    df['Vax_Dose3'] = pd.to_numeric(df['Vax_Dose3'], errors='coerce')
    df['Vax_Dose4'] = pd.to_numeric(df['Vax_Dose4'], errors='coerce')
    df['Vax_AllDoses'] = pd.to_numeric(df['Vax_AllDoses'], errors='coerce')

    #analyse the primary predictors
    primary_predictors(df, ['Araçatuba', 'Aracaju'])


"""
dfFull = pd.read_csv('3.1_3.2_Final_Dataset_State_Level_No_Missing.csv')
print(dfFull.shape, 'no missing')
##set data column to datetime objects
dfFull['Date'] = pd.to_datetime(dfFull['Date'])

# 1. Replace the string 'NA' with numpy's NaN
dfFull['Ct_Value'] = dfFull['Ct_Value'].replace('NA', np.nan)
# 2. Now check the count of NaN values
print(dfFull['Ct_Value'].isnull().sum())

# Convert Ct_Value to numeric, coercing any remaining non-numeric strings to NaN
dfFull['Ct_Value'] = pd.to_numeric(dfFull['Ct_Value'], errors='coerce')

# --- 1. Histogram of Ct_Value (Dropping Nulls for Visualization) ---
# Create a DataFrame for Ct_Value analysis by dropping all rows where Ct_Value is NaN
df_ct_clean = dfFull.dropna(subset=['Ct_Value']).copy()

# Create a Histogram of Ct_Values
plt.figure(figsize=(10, 6))
sns.histplot(df_ct_clean['Ct_Value'], bins=30, kde=True) ##kde=true -> adds the estimate smooth curve over top
plt.title('Distribution of Cycle Threshold (Ct) Values')
plt.xlabel('Ct Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('ct_value_histogram.png')
plt.close()

# --- 2. Time Series Plot for NewCases (Aggregation) ---
# Ensure NewCases is numeric before summing
dfFull['NewCases'] = pd.to_numeric(dfFull['NewCases'], errors='coerce') #errors='coerce': replaces non numeric values w NaN
# Aggregate NewCases by Date to get the national trend
df_national_cases = dfFull.groupby('Date')['NewCases'].sum().reset_index()

# Create a Time Series Plot for NewCases
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_national_cases, x='Date', y='NewCases') #plot nationalnewcases by date for num new cases
plt.title('National Daily New Cases Over Time')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('national_new_cases_time_series.png')
plt.close()
"""