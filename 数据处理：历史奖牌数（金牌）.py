import pandas as pd

# 读取CSV文件
df = pd.read_csv('summerOly_medal_counts.csv')

# 清除列名中的前后空格（如果有的话）
df.columns = df.columns.str.strip()

# 清除 NOC 列中的前后空格（如果有的话）
df['NOC'] = df['NOC'].str.strip()

# 清除其他可能存在空格的列的空格（例如年份、金牌数等）
df['Year'] = df['Year'].astype(str).str.strip()  # 年份列转换为字符串并去掉空格
df['Gold'] = df['Gold'].astype(str).str.strip()  # 金牌列去掉空格

df = df.dropna(subset=['Gold'])

# 提取CSV文件中实际出现的年份列表
unique_years = sorted(df['Year'].unique())

country_gold_medals = {}

for index, row in df.iterrows():
    country = row['NOC']
    gold_medals = row['Gold']
    year = row['Year']

    if pd.isna(gold_medals):
        continue

    if country not in country_gold_medals:
        country_gold_medals[country] = {}

    if year in country_gold_medals[country]:
        country_gold_medals[country][year] += int(gold_medals)
    else:
        country_gold_medals[country][year] = int(gold_medals)

result = []
for country, medals in country_gold_medals.items():
    gold_medals_info = []
    for year in unique_years:
        gold_count = medals.get(year, 0)
        gold_medals_info.append(str(gold_count))

    result.append(f"{country}: {' - '.join(gold_medals_info)}")

# 打印生成的列表
for item in result:
    print(item)

