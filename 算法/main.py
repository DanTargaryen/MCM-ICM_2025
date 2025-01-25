import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('results.csv')

# 1. 获取所有独立的NocSport（所有国家）
all_nocs = df['NocSport'].unique()

# 2. 获取1896到2020年数据
years_train = range(1896, 2021)
df_train = df[df['Year'].isin(years_train)]

# 3. 扩展数据：为每个年份和每个NocSport创建一个行，缺失的NocSport的所有特征置为零
expanded_data = []
for year in years_train:
    # 获取当前年份的所有NocSport
    year_data = df_train[df_train['Year'] == year]
    # 对于当前年份，所有的NocSport都需要出现在数据中
    for noc in all_nocs:
        if noc not in year_data['NocSport'].values:
            # 如果某个NocSport在当前年份没有出现，补充这一行
            expanded_data.append({
                'Year': year,
                'NocSport': noc,
                'Team': '',
                'PreGold': 0,
                'PreSilver': 0,
                'PreBronze': 0,
                'PreTotal': 0,
                'Participation': 0,
                'No_Medal': 0,
                'One_Medal': 0,
                'More_than_two': 0,
                'Host': 0,
                'Gold': 0
            })
        else:
            # 如果该NocSport在当前年份有数据，直接添加
            expanded_data.append(year_data[year_data['NocSport'] == noc].iloc[0].to_dict())

# 将扩展后的数据转换为DataFrame
expanded_df = pd.DataFrame(expanded_data)

# 4. 特征矩阵 X 和目标变量 Y
X_train = expanded_df[['NocSport', 'Year', 'Team', 'PreGold', 'PreSilver', 'PreBronze',
                       'PreTotal', 'Participation', 'No_Medal', 'More_than_two', 'Host']]
y_train = expanded_df['Gold']

# 5. 使用 OneHotEncoder 对类别特征进行编码
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['NocSport', 'Team', 'Year']),  # 对类别列进行 one-hot 编码
        ('num', 'passthrough', ['PreGold', 'PreSilver', 'PreBronze', 'PreTotal',
                                'Participation', 'No_Medal', 'More_than_two', 'Host'])  # 数值列保留
    ])

# 6. 创建和训练一个 Pipeline
model = LinearRegression()
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# 7. 拟合模型
pipeline.fit(X_train, y_train)

# 8. 使用训练好的模型计算2024年各个NocSport的金牌数
df_2024 = df[df['Year'] == 2024]  # 2024年的数据
X_2024 = df_2024[['NocSport', 'Year', 'Team', 'PreGold', 'PreSilver', 'PreBronze',
                  'PreTotal', 'Participation', 'No_Medal', 'More_than_two', 'Host']]

# 扩展2024年数据，确保每个NocSport都出现，并将缺失的NocSport的所有特征置为零
expanded_2024_data = []
for noc in all_nocs:
    if noc not in df_2024['NocSport'].values:
        expanded_2024_data.append({
            'Year': 2024,
            'NocSport': noc,
            'Team': '',
            'PreGold': 0,
            'PreSilver': 0,
            'PreBronze': 0,
            'PreTotal': 0,
            'Participation': 0,
            'No_Medal': 0,
            'One_Medal': 0,
            'More_than_two': 0,
            'Host': 0
        })
    else:
        expanded_2024_data.append(df_2024[df_2024['NocSport'] == noc].iloc[0].to_dict())

# 将扩展后的2024年数据转换为DataFrame
expanded_2024_df = pd.DataFrame(expanded_2024_data)

# 进行预测
X_2024 = expanded_2024_df[['NocSport', 'Year', 'Team', 'PreGold', 'PreSilver', 'PreBronze',
                           'PreTotal', 'Participation', 'No_Medal', 'More_than_two', 'Host']]

# 使用训练好的模型进行预测
y_pred_2024 = pipeline.predict(X_2024)

# 将预测值中小于零的部分置为零
y_pred_2024 = np.maximum(y_pred_2024, 0)

# 9. 获取2024年真实金牌数
y_true_2024 = expanded_2024_df['Gold'].values

# 10. 可视化预测值与真实值的对比
noc_sports_2024 = expanded_2024_df['NocSport'].values

# 获取原始2024年数据中的NocSport
original_nocs_2024 = df_2024['NocSport'].unique()

# 打印原数据中2024年的NocSport，且只打印实际存在的（没有填充的）数据
for noc_sport, pred_gold, true_gold in zip(noc_sports_2024, y_pred_2024, y_true_2024):
    if noc_sport in original_nocs_2024:  # 只打印原数据中2024年有的NocSport
        print(f"{noc_sport}: pred_gold = {pred_gold:.2f}, true_gold = {true_gold}")

# 输出回归系数和偏置项
print("回归系数 (K):", pipeline.named_steps['regressor'].coef_)
print("偏置项 (B):", pipeline.named_steps['regressor'].intercept_)

# 只保留原数据中的NocSport
filtered_noc_sports = []
filtered_pred_gold = []
filtered_true_gold = []

for noc_sport, pred_gold, true_gold in zip(noc_sports_2024, y_pred_2024, y_true_2024):
    if noc_sport in original_nocs_2024:  # 只保留原数据中2024年有的NocSport
        filtered_noc_sports.append(noc_sport)
        filtered_pred_gold.append(pred_gold)
        filtered_true_gold.append(true_gold)

# 计算均方误差 (MSE)
mse = mean_squared_error(filtered_true_gold, filtered_pred_gold)

# 计算均方根误差 (RMSE)
rmse = np.sqrt(mse)

# 计算平均绝对误差 (MAE)
mae = mean_absolute_error(filtered_true_gold, filtered_pred_gold)

# 计算决定系数 (R²)
r2 = r2_score(filtered_true_gold, filtered_pred_gold)

# 输出误差度量
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"决定系数 (R²): {r2:.2f}")

# 将预测金牌数与对应的Noc进行组合
predictions_2024 = pd.DataFrame({
    'Noc': expanded_2024_df['Noc'],  # 使用 'Noc' 列
    'PredictedGold': y_pred_2024
})

# 统计每个Noc的预测金牌总数
predicted_gold_per_noc_2024 = predictions_2024.groupby('Noc')['PredictedGold'].sum().reset_index()

# 打印每个Noc在2024年的预测金牌总数
print(predicted_gold_per_noc_2024)

# 将统计结果写入CSV文件
predicted_gold_per_noc_2024.to_csv('predicted_gold_per_noc_2024.csv', index=False)
