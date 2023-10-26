import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 1. 导入数据
datasets = pd.read_excel('/Users/yuanyuan/Desktop/ictis/ictisBS/siconc_D_2018.4.1-2023.4.1.xlsx')
time_series = datasets.iloc[:, 1].values

ts_length = 1827

# 2. 创建特征和标签
n_steps = 50
n_predictions = 7  # 预测未来10个时间步
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



# 计算评价指标
# 计算r^2
r_squared = r2_score(y_test, y_pred)
print("r^2:", r_squared)
print(' MAE : ', mae(y_test, y_pred))
print(' MAPE : ', mape(y_test, y_pred))
print(' RMSE : ', np.sqrt(mse(y_test, y_pred)))
mse = mean_squared_error(y_test, y_pred)
print('均方误差：', mse)

# 结果可视化
# 创建日期范围
date_range = pd.date_range(start='2022-4-06', end='2023-3-26', freq='D')#50出7（0）

# 创建一个新的图形和坐标轴对象
fig, ax = plt.subplots()

# 设置日期格式
date_format = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)
# 设置Y轴的刻度间隔
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
# 设置x轴和y轴的标签
ax.set_xlabel('time')
ax.set_ylabel('sea ice concentration')

# 绘制数据
ax.plot(date_range, y_test[:, 0], color='blue', label='ground truth')
ax.plot(date_range, y_pred[:, 0], color='red', label='predictions')
#y_test[:, -1]表示取 y_test 的所有行（即所有样本）的最后一个元素，也就是最后一个时间步的预测结果
# 显示图例
ax.legend()

# 显示图形
plt.show()

# 进行一次（十步）预测，，n_steps = 100
x_input = time_series[-n_steps:].reshape(1,-1)
y_hat = best_model.predict(x_input) #初始的第一次预测，得到一个10步的预测数据。

# 打印预测结果
print("预测值:", y_hat)
#print("真实值:", time_series[-6:])



