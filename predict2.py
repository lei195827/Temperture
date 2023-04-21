import torch
import pandas as pd
import numpy as np
from joblib import load
from temp2 import MLP
from sklearn.preprocessing import StandardScaler
from torch import nn


def load_model(model_path):
    """加载已训练好的PyTorch模型，并将其移动到可用的设备上"""
    model = MLP(input_size=7).cuda()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, device


def predict_temperature(model, device, input_data, scaler):
    """使用已经训练好的模型预测输入数据的温度值"""
    columns = ["CPU0 Temperature", "CPU1 Temperature", "CPU2 Temperature", "CPU3 Temperature", "module Humidity", "Module_temp", "lm75"]
    input_data_df = pd.DataFrame(input_data, columns=columns)
    input_data_scaled = scaler.transform(input_data_df)
    input_data_tensor = torch.tensor(input_data_scaled).float().to(device)
    output = model(input_data_tensor)
    output_np = output.detach().cpu().numpy()
    return output_np


if __name__ == '__main__':
    # 加载保存的scaler对象
    scaler = load('scaler.joblib')
    # 加载已经训练好的模型
    model, device = load_model('model.pth')

    # 预测输入数据的温度值
    input_data = np.array([38000, 39000, 37000, 36000, 38.228, 29.764, 343]).reshape(1, -1)
    output = predict_temperature(model, device, input_data, scaler)
    print(output)  # 22.934   23.32
