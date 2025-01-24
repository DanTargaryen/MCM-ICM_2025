import pandas as pd
import numpy as np

df = pd.read_csv('summerOly_athletes.csv')

df_2024 = df[df['Year'] == 2024]

countries = df_2024['Team'].unique()  # 所有国家
athletes_events = df_2024[['Name', 'Sport', 'Event']].drop_duplicates()  # 运动员-项目组合
medals = ['Gold', 'Silver', 'Bronze', 'No Medal']  # 奖牌类型

country_map = {country: idx for idx, country in enumerate(countries)}
athlete_event_map = {tuple(row): idx for idx, row in enumerate(athletes_events.values)}
medal_map = {medal: idx for idx, medal in enumerate(medals)}

matrix_3d = np.zeros((len(countries), len(athletes_events), len(medals)))

for _, row in df_2024.iterrows():
    country_idx = country_map[row['Team']]
    athlete_event_idx = athlete_event_map[(row['Name'], row['Sport'], row['Event'])]
    medal_idx = medal_map[row['Medal']] if row['Medal'] in medal_map else medal_map['No Medal']

    matrix_3d[country_idx, athlete_event_idx, medal_idx] += 1

# 打印示例：查看 China 国家，运动员 "A Dijiang" 在 "Basketball Men's Basketball" 项目中获得的奖牌情况
usa_idx = country_map['China']
athlete_event_idx = athlete_event_map[('A Dijiang', 'Basketball', "Basketball Men's Basketball")]

print(f"China, A Dijiang - Basketball Men's Basketball奖牌情况: {matrix_3d[usa_idx, athlete_event_idx]}")