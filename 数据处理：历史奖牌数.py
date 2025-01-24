import pandas as pd

# 读取CSV文件
df = pd.read_csv('summerOly_medal_counts.csv')

# 提取所需的列并生成新的列表
medal_list = df[['Year', 'NOC', 'Gold', 'Silver', 'Bronze', 'Total']]

# 将每一行数据格式化为'年份-国家-金牌数-银牌数-铜牌数-奖牌总数'
medal_info = []
for index, row in medal_list.iterrows():
    medal_info.append(f"{row['Year']}-{row['NOC']}-{row['Gold']}-{row['Silver']}-{row['Bronze']}-{row['Total']}")

# 打印生成的列表
for info in medal_info:
    print(info)