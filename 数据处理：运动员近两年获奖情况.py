import pandas as pd

# 读取CSV文件
df = pd.read_csv('summerOly_athletes.csv')

# 去除列名中的前后空格
df.columns = df.columns.str.strip()

# 清除缺失或不合规的奖牌记录
df = df[df['Medal'].notna()]
df = df[df['Medal'].isin(['Gold', 'Silver', 'Bronze'])]

# 过滤数据，提取2020年和2024年的数据
df_filtered = df[df['Year'].isin([2020, 2024])]

# 初始化一个字典来存储每个运动员的相关数据
athlete_medals = {}

# 遍历每一行数据
for index, row in df_filtered.iterrows():
    key = (row['NOC'], row['Name'], row['Sport'], row['Event'])  # 使用国家、运动员姓名、运动、项目作为唯一标识
    if key not in athlete_medals:
        athlete_medals[key] = {
            'gold_2020': 0, 'silver_2020': 0, 'bronze_2020': 0, 'total_2020': 0,
            'gold_2024': 0, 'silver_2024': 0, 'bronze_2024': 0, 'total_2024': 0
        }

    # 根据年份更新相应奖牌类型的计数
    if row['Year'] == 2020:
        if row['Medal'] == 'Gold':
            athlete_medals[key]['gold_2020'] += 1
        elif row['Medal'] == 'Silver':
            athlete_medals[key]['silver_2020'] += 1
        elif row['Medal'] == 'Bronze':
            athlete_medals[key]['bronze_2020'] += 1
        athlete_medals[key]['total_2020'] += 1
    elif row['Year'] == 2024:
        if row['Medal'] == 'Gold':
            athlete_medals[key]['gold_2024'] += 1
        elif row['Medal'] == 'Silver':
            athlete_medals[key]['silver_2024'] += 1
        elif row['Medal'] == 'Bronze':
            athlete_medals[key]['bronze_2024'] += 1
        athlete_medals[key]['total_2024'] += 1

# 生成最终的列表
result = []
for key, medals in athlete_medals.items():
    country, name, sport, event = key
    total_gold = medals['gold_2020'] + medals['gold_2024']
    total_silver = medals['silver_2020'] + medals['silver_2024']
    total_bronze = medals['bronze_2020'] + medals['bronze_2024']
    total = total_gold + total_silver + total_bronze  # 确保总奖牌数是金银铜的总和

    result.append(
        f"{country}-{name}-{sport} & {event}-{total_gold}-{total_silver}-{total_bronze}-{total}")

# 打印生成的列表
for item in result:
    print(item)

