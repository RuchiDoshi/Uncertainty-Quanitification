import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('2.1_2.2_Final_Dataset_City_Level.xlsx', sheet_name = '2.1_City_Daily_Missing')
df.to_csv('2.1_City_Daily_Missing.csv', index=False)