import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('results(6).csv')

numeric_columns = df.select_dtypes(include=['number']).columns  # 选择数值类型的列
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())  # 填补缺失值

df['Year'] = df['Year'].astype(int)
df['Host'] = df['Host'].astype(int)

# 选择预测金牌数的特征
features = ['Year', 'PreGold', 'PreSilver', 'PreBronze', 'PreTotal', 'Participation', 'No_Medal', 'More_than_two', 'Host']
target = 'Gold'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#构建随机森林模型
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

feature_importances = pd.DataFrame(rf.feature_importances_, index=features, columns=['Importance'])
print(feature_importances)

df_2024 = df[df['Year'] == 2024]
X_2024 = df_2024[features]
X_2024_scaled = scaler.transform(X_2024)

predictions_2024 = rf.predict(X_2024_scaled)

df_2024.loc[:, 'Predicted_Gold'] = predictions_2024  # 使用 .loc 避免 SettingWithCopyWarning

df_2024[['NocSport', 'Predicted_Gold']].to_csv('predicted_2024_gold_medals.csv', index=False)

print("2024年奥运会金牌预测已导出为 CSV 文件。")


