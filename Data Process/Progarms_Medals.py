import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('summerOly_athletes.csv')

def getMedalNumber(df,year,country,program):
    for index,row in df.iterrows():
        if row["NOC"]==country and row["Year"]==year and row["Sport"]==program:
            return row["Medal"]

#项目列表
Program_list = []
for program in df['Sport']:
    if program not in Program_list:
        Program_list.append(program)

#年份列表
Year_list = []
for year in df['Year']:
    if year not in Year_list:
        Year_list.append(year)

#国家列表
Country_list = []
for country in df['NOC']:
    if country not in Country_list:
        Country_list.append(country)

#国家-年份-项目详细表
Program_Medals = []
for country in Country_list:
    countryList = []
    for year in Year_list:
        yearList = []
        for program in Program_list:
            yearList.append(getMedalNumber(df,year,country,program))
        countryList.append(yearList)
    Program_Medals.append(countryList)

print(Program_Medals[:1])


