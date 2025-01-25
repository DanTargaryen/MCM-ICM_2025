import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
df = pd.read_csv('summerOly_athletes.csv')

# 创建 NocSport 和 Year 列的唯一组合
Noc_list = []
for _, row in df.iterrows():
    if [f"{row['NOC']}-{row['Sport']}", f"{row['Year']}"] not in Noc_list:
        Noc_list.append([f"{row['NOC']}-{row['Sport']}", f"{row['Year']}"])

# 创建一个新的 DataFrame df_new
df_new = pd.DataFrame(Noc_list, columns=['NocSport', 'Year'])

# 添加 NocSport 列到原 DataFrame
df['NocSport'] = df['NOC'] + '-' + df['Sport']
df.drop(['Sport', 'Sex', 'NOC', 'Event', 'Team', 'City'], axis=1, inplace=True)

# 按 NocSport 列分组
grouped = df.groupby('NocSport')

# 使用 loc 更新 PreGold、PreSilver 和 PreBronze
df_new['PreGold'] = 0
df_new['PreSilver'] = 0
df_new['PreBronze'] = 0
df_new['PreTotal'] = 0
df_new['Participation'] = 0
df_new['No_Medal'] = 0
df_new['One_Medal'] = 0
df_new['More_than_two'] = 0

for index, row in df_new.iterrows():
    year = int(row['Year'])
    grouped_row = grouped.get_group(row['NocSport'])
    
    # 计算 PreGold, PreSilver, PreBronze
    df_new.loc[index, 'PreGold'] = grouped_row[(grouped_row['Year'] < year) & (grouped_row['Medal'] == 'Gold')].shape[0]
    df_new.loc[index, 'PreSilver'] = grouped_row[(grouped_row['Year'] < year) & (grouped_row['Medal'] == 'Silver')].shape[0]
    df_new.loc[index, 'PreBronze'] = grouped_row[(grouped_row['Year'] < year) & (grouped_row['Medal'] == 'Bronze')].shape[0]
    
    # 计算 PreTotal
    df_new.loc[index, 'PreTotal'] = df_new.loc[index, 'PreGold'] + df_new.loc[index, 'PreSilver'] + df_new.loc[index, 'PreBronze']
    
    # 计算 Participation
    df_new.loc[index, 'Participation'] = grouped_row[grouped_row['Year'] == year].shape[0]

    grouped_by_name = grouped_row.groupby('Name')
    Count_None = 0
    Count_One = 0
    Count_MorethanTwo = 0
    for name,group in grouped_by_name:
        if group[group['Medal'] != 'No medal'].shape[0]>=2:
            Count_MorethanTwo += 1
        elif group[group['Medal'] != 'No medal'].shape[0]==1:
            Count_One += 1
        else:
            Count_None += 1

    df_new.loc[index, 'No_Medal'] = Count_None
    df_new.loc[index, 'One_Medal'] = Count_One
    df_new.loc[index, 'More_than_two'] = Count_MorethanTwo


df_new_sorted = df_new.sort_values(by='NocSport')

# 导出排序后的 DataFrame 为 CSV 文件
df_new_sorted.to_csv('results.csv', index=False)

# 如果需要显示导出成功的消息
print("Data has been successfully exported to 'results.csv'.")

