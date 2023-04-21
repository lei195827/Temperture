import os

import torch
import pandas as pd
import numpy as np
from joblib import load
from temp2 import MLP
import itertools
from sklearn.preprocessing import StandardScaler
from torch import nn


def load_test_data_mlp(test_xls, scaler, device, column_xls):
    df_test = pd.read_excel(column_xls, sheet_name="Sheet1")
    test_data = pd.read_excel(test_xls, sheet_name="data")
    test_data = test_data.loc[:,
                ~test_data.columns.isin([' Brightness', ' Show Time', " fixed humidity", ' fixed temperature'])]
    test_data = pd.concat([test_data, df_test])
    test_data = test_data.fillna(0)
    print(test_data.columns)

    print(test_data)
    input_date = test_data.loc[:, ~test_data.columns.isin(["GT_Temp"])]
    input_data_scaled = scaler.transform(input_date)

    input_data_tensor = torch.tensor(input_data_scaled).float().to(device)
    return input_data_tensor


def load_model_mlp(model_path, input_size=8):
    """加载已训练好的PyTorch模型，并将其移动到可用的设备上"""
    model = MLP(input_size=input_size).cuda()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, device


def add_predict_data(test_xls, output_data_list):
    # 分割文件名和扩展名
    name, ext = os.path.splitext(test_xls)
    # 读取xls文件为DataFrame对象
    df = pd.read_excel(test_xls)
    # 给DataFrame对象添加新列，赋值为一个列表
    output_data_list = list(itertools.chain(*output_data_list))
    output_data_iter = iter(output_data_list)
    for i in range(len(df)):
        df.loc[i, 'predict_data'] = next(output_data_iter)
    # 把修改后的DataFrame对象保存回xls文件
    new_name = name + '_pre' + ext
    df.to_excel(new_name)


def predict_temperature_mlp(model, input_datas):
    """使用已经训练好的模型预测输入数据的温度值"""
    output_data_list = []
    for row in input_datas:
        input_data = row.reshape(1, -1)
        output = model(input_data)
        output_np = output.detach().cpu().numpy()
        output_data_list.append(output_np)
    for output_data in output_data_list:
        print(output_data)
    return output_data_list


if __name__ == '__main__':
    input_size = 8
    # 加载保存的scaler对象
    scaler = load('scaler.joblib')
    # 加载已经训练好的模型
    model, device = load_model_mlp('model_mlp.pth', input_size=input_size)
    test_xls = "test/10.xls"
    input_data_tensor = load_test_data_mlp(test_xls, scaler, device, column_xls='columns.xlsx')
    output_data_list = predict_temperature_mlp(model, input_data_tensor)

    add_predict_data(test_xls=test_xls, output_data_list=output_data_list)
