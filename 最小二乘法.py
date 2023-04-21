import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import dump
from temp2 import get_number

# 读取文件路径和表名
train_folder = "train1/"
test_xls = "test/13.xls"
ori_train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if
                   os.path.isfile(os.path.join(train_folder, f)) and os.path.splitext(f)[1] == ".xls"]
train_files = sorted(ori_train_files, key=get_number)
print(train_files)
# 将所有数据拼接成一个 DataFrame
train_data = pd.DataFrame()
for file in train_files:
    if os.path.basename(file) != os.path.basename(test_xls):
        df = pd.read_excel(file, sheet_name="data", skiprows=list(range(1, 300)))
        df = df.loc[:, ~df.columns.isin([' Brightness', ' Show Time', " fixed humidity", ' fixed temperature'])]
        train_data = pd.concat([train_data, df])
    else:
        print(f'test xls is:{file}')

df_test = pd.DataFrame(columns=train_data.columns)
df_test.to_excel('columns.xlsx', index=False)
test_data = pd.read_excel(test_xls, sheet_name="data")
train_data = train_data.loc[:,
             ~train_data.columns.isin([' Brightness', ' Show Time', " fixed humidity", ' fixed temperature'])]
test_data = test_data.loc[:,
            ~test_data.columns.isin([' Brightness', ' Show Time', " fixed humidity", ' fixed temperature'])]
test_data = pd.concat([test_data, df_test])

train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
print("train_data and test_data shape", train_data.shape, test_data.shape)
# 将数据标准化
scaler = StandardScaler()
scaler.feature_names = ["CPU0_T", "CPU1_T", "CPU2_T", "CPU3_T", "Module_humity", "Module_temp", "lm75",
                        ' IsTalking']
all_feature = train_data.loc[:, ~train_data.columns.isin(["GT_Temp"])]
test_feature = test_data.loc[:, ~test_data.columns.isin(["GT_Temp"])]
print(type(all_feature))
print(all_feature.shape)

labels = train_data["GT_Temp"]
test_labels = test_data["GT_Temp"]


all_feature = scaler.fit_transform(all_feature)
# 保存scaler对象到文件
dump(scaler, 'scaler_lsm.joblib')
test_feature = scaler.transform(test_feature)

# 构造训练集和测试集的自变量和因变量
x_train = sm.add_constant(all_feature)
y_train = labels
x_test = sm.add_constant(test_feature)
y_test = test_labels

# 构建线性回归模型并训练
model = sm.OLS(y_train, x_train)
result = model.fit()

# 在测试集上进行预测并计算MSE
y_pred = result.predict(x_test)
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error (MSE): ", mse)

# 可视化预测结果和原始值
plt.plot(test_data.index, y_test, label='GT_Temp')
plt.plot(test_data.index, y_pred, label='Predicted GT_Temp')
plt.legend()
plt.show()
