import torch
import numpy as np
import pandas as pd
from torch import nn
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from temp2 import MLP


# 定义神经网络模型
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16 + input_size, 8)
        self.out = nn.Linear(8 + input_size, 1)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(input_size)

    def forward(self, x):
        x = self.bn3(x)
        x1 = self.dp(self.bn1(self.relu(self.fc1(x))))
        x2 = torch.cat([x1, x], axis=1)
        x3 = self.dp(self.bn2(self.relu(self.fc2(x2))))
        x3 = torch.cat([x3, x], axis=1)
        xout = self.out(x3)
        return xout


def predict_temperature(input_data):
    # 加载保存的模型
    model = torch.load_state_dict(torch.load("model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 设置模型为评估模式
    model.eval()
    # 传递输入数据进行预测
    columns = ['Temperature', ' module Humidity', ' module temperature', ' lm75']
    # 创建StandardScaler对象
    scaler = StandardScaler()
    # 加载scaler对象
    scaler = load('scaler.joblib')
    # 将数据转换为DataFrame，并设置列名
    input_data_df = pd.DataFrame(input_data, columns=columns)
    print(input_data_df)
    input_data_scaled = scaler.transform(input_data_df)
    print(input_data_scaled)
    input_data_tensor = torch.from_numpy(input_data_scaled)
    # 将标准化后的数据转换为tensor，并移动到GPU上
    input_data_tensor = torch.tensor(input_data_scaled).float().to(device)
    output = model(input_data_tensor)
    # 将输出转换为numpy数组并返回
    output_np = output.cpu().detach().numpy()
    return output_np


if __name__ == '__main__':
    input_data = np.array([38000, 39000, 37000, 36000, 38.228, 29.764, 343]).reshape(1, -1)
    # input_data = np.array([39000,	39000,	38000,	37000,	40.322,	29.905,	351]).reshape(1, -1)
    result = input_data[:, :4].sum(axis=1)
    input_data = np.hstack((result.reshape((-1, 1)), input_data[:, 4:]))
    output = predict_temperature(input_data)
    print(output)  # 22.934   23.32
