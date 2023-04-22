import os

import torch
import pandas as pd
import numpy as np
from joblib import load
from matplotlib import pyplot as plt

from temp2 import MLP
import itertools
from sklearn.preprocessing import StandardScaler
from torch import nn

from temp_rnn_2 import RNN, RNN_2, LSTM, GRU


def load_test_data_rnn(test_xls, scaler, device, column_xls, sequence_length=50):
    df_test = pd.read_excel(column_xls, sheet_name="Sheet1")
    test_data = pd.read_excel(test_xls, sheet_name="data")
    test_data = test_data.loc[:,
                ~test_data.columns.isin([' Brightness', ' Show Time', " fixed humidity", ' fixed temperature'])]
    test_data = pd.concat([test_data, df_test])
    test_data = test_data.fillna(0)
    print(test_data.columns)
    output = test_data["GT_Temp"]
    input_tensor_rnn = []
    print(test_data)
    input_date = test_data.loc[:, ~test_data.columns.isin(["GT_Temp"])]
    input_data_scaled = scaler.transform(input_date)
    input_data_scaled = np.array(input_data_scaled)  # add this line
    for i in range(len(input_data_scaled) - sequence_length):
        input_tensor_rnn.append(input_data_scaled[i:i + sequence_length])
    input_tensor_rnn = torch.tensor(input_tensor_rnn).float().to(device)
    return input_tensor_rnn, output


def load_model_rnn(model_rnn_path, input_size=8, hidden_size=16, num_layers=1, output_size=1, fc_size=8, net=RNN_2):
    """加载已训练好的PyTorch模型，并将其移动到可用的设备上"""
    # 创建RNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_rnn = net(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
                    , output_size=output_size, fc_size=fc_size).to(device)
    # model = RNN_2(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
    #               , output_size=output_size,fc_size=fc_size).cuda()
    model_rnn.load_state_dict(torch.load(model_rnn_path))


    model_rnn.eval()
    return model_rnn, device


def predict_temperature_rnn(model, input_tensor_rnn, sequence_length=50, input_size=8):
    """使用已经训练好的模型预测输入数据的温度值"""
    output_data_list = []
    with torch.no_grad():
        for input_data in input_tensor_rnn:
            input_data = input_data.view(-1, sequence_length, input_size)
            output_rnn = model(input_data)
            output_data_list.append(output_rnn.detach().cpu().numpy())

    for output_data in output_data_list:
        print(output_data)
    return output_data_list


def add_predict_data(test_xls, output_data_list, sequence_length=50):
    # 分割文件名和扩展名
    name, ext = os.path.splitext(test_xls)
    # 读取xls文件为DataFrame对象
    df = pd.read_excel(test_xls)
    # 给DataFrame对象添加新列，赋值为一个列表
    output_data_list = list(itertools.chain(*output_data_list))
    # 从第50个开始写入
    output_data_iter = iter(output_data_list)
    for i in range(sequence_length, len(df)):
        df.loc[i, 'predict_data'] = next(output_data_iter)
    # 把修改后的DataFrame对象保存回xls文件
    new_name = name + '_pre' + ext
    df.to_excel(new_name)


if __name__ == '__main__':
    # 加载保存的scaler对象
    scaler = load('scaler.joblib')
    sequence_length = 50
    input_size = 8
    hidden_size = 8
    fc_size = 8
    num_layers = 2
    output_size = 1
    # 加载已经训练好的模型
    model_rnn, device = load_model_rnn(model_rnn_path='model_gru.pth', input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers, fc_size=fc_size,
                                       output_size=output_size, net=GRU)
    test_xls = "test/11.xls"

    input_tensor_rnn, output = load_test_data_rnn(test_xls, scaler, device, column_xls='columns.xlsx',
                                                  sequence_length=sequence_length)
    output_data_list = predict_temperature_rnn(model_rnn, input_tensor_rnn,
                                               sequence_length=sequence_length,
                                               input_size=input_size)
    # 可视化预测结果和原始值
    index = range(len(output_data_list))
    np_out = np.asarray(output_data_list).reshape(-1)
    plt.plot(index, np_out, label='GT_Temp')
    plt.plot(index, output[sequence_length - 1:-1], label='Predicted GT_Temp')
    plt.legend()
    plt.show()
    add_predict_data(test_xls=test_xls, output_data_list=output_data_list, sequence_length=sequence_length)
