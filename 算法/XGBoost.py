import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from scipy.stats import uniform, randint


scaler1 = StandardScaler()
scaler2 = StandardScaler()

train_data = pd.read_csv("results.csv")

#去除名字、国别等
train_data = train_data.drop(['NocSport','Team'],axis=1)

#划分特征和标签
X_data = train_data.iloc[:,:-1]
Y_data = train_data.iloc[:,-1:]

#预处理
X_data = scaler1.fit_transform(X_data)
Y_data = scaler2.fit_transform(Y_data)

#输入特征数量
train_features = X_data.shape[1]

#划分数据集
X_train,X_test,Y_train,Y_test = train_test_split(X_data,Y_data,test_size=0.3)

#回归模型
xgb_regressor = xgb.XGBRFRegressor(
    objective = 'reg:squarederror',
    random_state = 42,
    )

#参数设置
param_dist = {
'n_estimators': randint(100, 1000),
'learning_rate': uniform(0.01, 0.19),  # 从0.01到0.2
'max_depth': randint(3, 10),           # 从3到10
'min_child_weight': randint(1, 10),
'gamma': uniform(0, 0.5),
'subsample': uniform(0.5, 0.5),        # 从0.5到1.0
'colsample_bytree': uniform(0.5, 0.5),
'reg_alpha': uniform(0, 1),
'reg_lambda': uniform(1, 4)            # 从1到5
}

# 创建RandomizedSearchCV对象
random_search = RandomizedSearchCV(estimator=xgb_regressor, param_distributions=param_dist,n_iter=100,scoring='neg_mean_squared_error',
cv=3,
verbose=2,
random_state=42,
n_jobs=-1)

# 创建RandomizedSearchCV对象
random_search.fit(X_train,Y_train)

# 获取最佳参数
best_params = random_search.best_params_
print("最佳参数组合:", best_params)

# 获取最佳模型
best_model = random_search.best_estimator_

# 使用最佳模型进行预测
y_pred = best_model.predict(X_data)

#逆归一化
y_pred = np.reshape(y_pred,(-1,1))
y_pred = scaler2.inverse_transform(y_pred)
print(y_pred)
Y_data = scaler2.inverse_transform(Y_data)
print(Y_data)

#评价模型表现
mse = mean_squared_error(Y_data, y_pred)
mae = mean_absolute_error(Y_data, y_pred)
r2 = r2_score(Y_data,y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")


y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Predictions']
data_combined = pd.concat([train_data,y_pred],axis=1)
# 假设 df_combined 是你合并后的 DataFrame
data_combined.to_csv('merged_results.csv', index=False)

# 输出提示消息
print("Data has been successfully exported to 'merged_results.csv'.")

# 输出特征重要性
import matplotlib.pyplot as plt

# 训练模型
xgb_regressor.fit(X_train, Y_train)

xgb.plot_importance(xgb_regressor)
plt.show()