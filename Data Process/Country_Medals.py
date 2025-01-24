import pandas as pd

def getMedalNumber(data,Year,Country,medal):
    for index,row in data.iterrows():
        if row["NOC"] == country and row["Year"] == year:
            return row[f'{medal}']

# 读取CSV文件
df = pd.read_csv('summerOly_medal_counts.csv')

#国家列表
Country_list = []
for country in df['NOC']:
    if country not in Country_list:
        Country_list.append(country)
# print(Country_list)

#年份列表
Year_list = []
for year in df['Year']:
    if year not in Year_list:
        Year_list.append(year)
# print(Year_list)

#读取国家-年份列表
medal = 'Gold'
Country_Medals = []
for country in Country_list:
    temp_list = []
    for year in Year_list:
        temp_list.append(getMedalNumber(df,year,country,medal))
    Country_Medals.append(temp_list)

print(Country_Medals[:5])
