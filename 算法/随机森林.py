import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("results(4).csv")

# 检查数据中是否包含2024年
print(df['Year'].unique())

df_2024 = df[df['Year'] == 2024]

print(f"Number of rows for 2024 data: {df_2024.shape[0]}")
if df_2024.empty:
    print("No data for the year 2024.")
else:
    df_train = df[df['Year'] < 2024]

    features = ['PreGold', 'PreSilver', 'PreBronze', 'PreTotal', 'Participation', 'No_Medal', 'One_Medal', 'More_than_two', 'Host']
    target = 'Gold'

    df_train = df_train.dropna(subset=features + [target])

    X = df_train[features]
    y = df_train[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],  # 树的数量
        'max_depth': [10, 20, 30, None],  # 树的最大深度
        'min_samples_split': [2, 5, 10],  # 划分节点的最小样本数
        'min_samples_leaf': [1, 2, 4],    # 叶节点的最小样本数
        'max_features': ['sqrt', 'log2', None],  # 每次分裂考虑的最大特征数
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')

    # 训练模型
    grid_search.fit(X_train, y_train)

    # 打印最佳参数
    print("Best Parameters:", grid_search.best_params_)

    best_rf = grid_search.best_estimator_

    #在验证集上评估模型
    y_pred = best_rf.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f'Mean Absolute Error on validation set: {mae}')

    if not df_2024.empty:
        X_2024 = df_2024[features]
        X_2024_scaled = scaler.transform(X_2024)

        y_2024_pred = best_rf.predict(X_2024_scaled)

        # 将预测结果添加到2024年数据中
        df_2024.loc[:, 'Predicted_Gold'] = y_2024_pred  # 使用 .loc 来避免 SettingWithCopyWarning

        # 显示预测结果
        print(df_2024[['NocSport', 'Predicted_Gold']])

        # 将预测结果导出为CSV文件
        df_2024.to_csv("predicted_2024_gold.csv", index=False)
        print("Predicted data saved to 'predicted_2024_gold.csv'.")



