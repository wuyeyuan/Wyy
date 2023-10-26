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
import openpyxl
import warnings

warnings.filterwarnings('ignore')



# 1导入数据
datasets = pd.read_excel('/Users/yuanyuan/Desktop/ictisBS/siconc_D_2018.4.1-2023.4.1.xlsx')
time_series = datasets.iloc[:, 1].values#.reshape(-1, 1)

ts_length = 1827#数据个数


## 创建特征和标签：对于每个时间步，我们将过去的时间步作为特征，将未来的时间步作为标签，以便模型学习如何根据过去的时间步预测未来的时间步。
n_steps = 200
X = []
y = []
for i in range(n_steps, ts_length):
    X.append(time_series[i - n_steps:i].reshape(-1))
    y = np.concatenate((y, np.array([time_series[i - n_steps:i + 1][-1]])), axis=0)
#X、y为ts_length-n_steps=1577个。再划分训练集测试集
X = np.array(X)
y = np.array(y)



# 划分数据集，将后20%（316）划分为测试集，前80%（1261）划分为训练集。

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
study.optimize(objective, n_trials=100)#100

# 输出最佳超参数
print('Best trial:', study.best_trial.params)

# 使用最佳超参数进行预测
best_params = study.best_trial.params
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
#y_pred = best_model.predict(X_test)


# 计算评价指标
# 计算r^2
r_squared = r2_score(y_test, y_pred)
print("r^2:", r_squared)
print(' MAE : ', mae(y_test, y_pred))
print(' MAPE : ', mape(y_test, y_pred))
print(' RMSE : ', np.sqrt(mse(y_test, y_pred)))
mse = mean_squared_error(y_test, y_pred)
print('均方误差：', mse)

# 进行7步预测，n_steps = 6
x_input = time_series[-n_steps:].reshape(1,-1)
y_hat = best_model.predict(x_input) #初始的第一步预测，得到一个预测数据，下面循环y-x次，预测了y-x步，所以最后得到的y_hat预测了y-x+1步。
for i in range(1, 213): # range(x,y) =y-x 个
    x_input = np.append(x_input[:,1:], y_hat[-1]).reshape(1,-1)
    y_hat = np.append(y_hat, best_model.predict(x_input))



#在循环的每一次，首先将 x_input 的最左边一列去掉，再将上一步的预测结果 y_hat[-1] 添加到 x_input 最右侧的位置。
# 这样 x_input 会不断地向右移动，新的预测结果也会不断地添加到 y_hat 的末尾。
#接下来，调用 model.predict(x_input) 对移动后的 x_input 进行预测，将预测结果添加到 y_hat 的末尾。

# 它从时间序列数据中选择了最近的n_steps个数据点作为模型的输入。这些数据点被reshape成一个1行n_steps列的矩阵，作为模型的输入x_input。
# 然后，使用best_model对x_input进行预测，并将结果存储在y_hat中。
# 接下来，对于每一个时间步i，将y_hat的最后一个预测值添加到x_input的末尾，然后将其重新reshape成一个1行n_steps列的矩阵。
# 最后，再次使用best_model对新的x_input进行预测，并将预测结果添加到y_hat中。
# 通过这种方式，模型的预测值被迭代地添加到x_input中，从而得到了n_steps个未来时间步的预测值。

# 创建一个新的工作簿
wb = openpyxl.Workbook()

# 选择活动工作表
ws = wb.active

# 将预测值写入工作表
for i in range(len(y_hat)):
     ws.cell(row=i+1, column=1, value=y_hat[i])

# 保存工作簿
wb.save("/Users/yuanyuan/Desktop/ictisBS/多步预测213/y-BS-200-1-（213）.xlsx")

# 打印预测结果
#print("预测值:", y_hat)
#print("真实值:", time_series[-6:])

# 结果可视化
plt.figure(1)
#x = np.arange(1, 73)
#plt.plot(x[65:], y_hat, color='green', label='forecast')
plt.plot(y_pred, color='red', label='predict')
plt.plot(y_test, color='blue', label='true')
plt.title('Result visualization')
plt.legend()
plt.show()

