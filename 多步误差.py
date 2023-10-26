import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 1. 导入数据
datasets = pd.read_excel('/Users/yuanyuan/Desktop/ictisBS/siconc_D_2018.4.1-2023.4.1.xlsx')
time_series = datasets.iloc[:, 1].values

ts_length = 1827

# 2. 创建特征和标签
n_steps = 50
n_predictions = 10  # 预测未来10个时间步
X = []
y = []
for i in range(n_steps, ts_length - n_predictions+1):# 要加上1，X_test、y_test 才能遍历到序列最后一个
    X.append(time_series[i - n_steps:i].reshape(-1))
    y.append(time_series[i:i + n_predictions])

X = np.array(X)
y = np.array(y)


# 划分训练集和测试集
train_size = int(len(X) * 0.8)  # 取前80%作为训练集
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]


def objective(trial):
    # 定义超参数空间
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'max_depth': trial.suggest_int('max_depth', 3, 10, 12),
        'eta': trial.suggest_loguniform('eta', 1e-3, 1),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 1.0, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
        'n_estimators': trial.suggest_categorical('n_estimators', [800, 1000, 1200, 2000])
    }

    # 定义模型
    model = xgb.XGBRegressor(**params)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算 RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return rmse


# 创建 study 对象
study = optuna.create_study(direction='minimize')

# 运行优化
study.optimize(objective, n_trials=10)#100

# 输出最佳超参数
print('Best trial:', study.best_trial.params)

# 使用最佳超参数进行预测
best_params = study.best_trial.params
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
#y_pred = best_model.predict(X_test)

# 计算每个预测时间步的误差
#step_errors = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

# 绘制误差折线图
#plt.plot(step_errors)
#plt.ylabel('误差')
#plt.title('多步预测误差折线图')
#plt.show()

# 计算评价指标
r_squared = r2_score(y_test, y_pred)
print("r^2:", r_squared)
print(' MAE : ', mae(y_test[1, :], y_pred[1, :]))
#print(' MAPE : ', mape(y_test[1, :], y_pred[1, :]))
#print(' RMSE : ', np.sqrt(mse(y_test[1, :], y_pred[1, :])))
mse = mean_squared_error(y_test[1, :], y_pred[1, :])
print('均方误差：', mse)



# 结果可视化
plt.figure(1)
#x = np.arange(1, 67)
#plt.plot(x[59:], y_hat[0], color='green', label='forecast')
plt.plot(y_test[1, :],color='blue', label='True values')
#plt.plot(y_pred,color='red', label='Predicted values') #预测一步
plt.plot(y_pred[1, :],color='red', label='Predicted values') #预测多步
#y_test[:, -1]表示取 y_test 的所有行（即所有样本）的最后一个元素，也就是最后一个时间步的预测结果
#y_test[1, :]表示取 y_test 第一行，所有列的元素.
plt.title('Result visualization')
plt.legend()
plt.show()

