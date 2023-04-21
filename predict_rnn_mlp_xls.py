# encoding:utf-8
import os
import pandas as pd
import predict_rnn_xls
import predict_xls_mlp
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from pathlib import Path


def add_predict_data(test_xls, output_data_list):
    # 分割文件名和扩展名
    name, ext = os.path.splitext(test_xls)
    # 读取xls文件为DataFrame对象
    df = pd.read_excel(test_xls)
    # 给DataFrame对象添加新列，赋值为一个列表
    for i in range(len(df)):
        df.loc[i, 'predict_data'] = output_data_list[i]
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
    test_xls = "test/28.xls"
    # 加载已经训练好的模型
    model_mlp, device = predict_xls_mlp.load_model_mlp('model_mlp.pth', input_size=input_size)
    model_rnn, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_rnn.pth', input_size=input_size,
                                                       hidden_size=hidden_size,
                                                       num_layers=num_layers, fc_size=fc_size,
                                                       output_size=output_size)
    input_data_mlp = predict_xls_mlp.load_test_data_mlp(test_xls, scaler, device, column_xls='columns.xlsx')
    input_tensor_rnn, output = predict_rnn_xls.load_test_data_rnn(test_xls, scaler, device, column_xls='columns.xlsx',
                                                                  sequence_length=sequence_length)
    mlp_out = predict_xls_mlp.predict_temperature_mlp(model_mlp, input_data_mlp)
    rnn_out = predict_rnn_xls.predict_temperature_rnn(model_rnn, input_tensor_rnn,
                                                      sequence_length=sequence_length,
                                                      input_size=input_size)
    # 可视化预测结果和原始值
    plot_length = 8000
    index_rnn = range(sequence_length - 1, min(plot_length, len(rnn_out) + sequence_length - 1))
    index_mlp = range(min(plot_length, len(mlp_out)))
    np_mlp_out_all = np.asarray(mlp_out).reshape(-1)
    np_rnn_out_all = np.asarray(rnn_out).reshape(-1)

    np_mlp_out = np.asarray(mlp_out[:plot_length]).reshape(-1)
    np_rnn_out = np.asarray(rnn_out[:plot_length]).reshape(-1)
    mlp_ratio = 0.3
    rnn_ratio = 0.7
    hybrid_out = np.zeros(len(np_mlp_out_all))
    hybrid_out[:sequence_length] = np_mlp_out[:sequence_length]
    hybrid_out[sequence_length:] = rnn_ratio * np_rnn_out_all[:] + mlp_ratio * np_mlp_out_all[sequence_length:]
    # 计算MLP和RNN在预测结果中所占的比例

    plt.plot(index_rnn, np_rnn_out[:plot_length - (sequence_length - 1)], label='RNN_GT_Temp')
    if np.any(output):
        plt.plot(index_mlp, output[:plot_length], label='GT_Temp')
    else:
        print("all of output if '0'")
    plt.plot(index_mlp, np_mlp_out[:], label='MLP_GT_Temp')
    plt.plot(index_mlp, hybrid_out[:plot_length], label='Hybrid_GT_Temp')
    plt.legend()
    plt.savefig(os.path.join("result", Path(test_xls).stem + ".png"), dpi=300, bbox_inches='tight')  # 保存图片为png格式，分辨率为300，裁剪掉多余的空白
    plt.show()

    # 计算每个预测数组和其对应的真实值数组之间的均方根损失
    mlp_rmse = mean_squared_error(output, np_mlp_out_all, squared=False)
    rnn_rmse = mean_squared_error(output[sequence_length:], np_rnn_out_all,
                                  squared=False)
    hybrid_rmse = mean_squared_error(output[:], hybrid_out[:], squared=False)

    # 打印均方根损失
    print(f'MLP RMSE:{mlp_rmse:.3f}')
    print(f'RNN RMSE:{rnn_rmse:.3f}')
    print(f'Hybrid RMSE:{hybrid_rmse:.3f}')

    # add_predict_data(test_xls=test_xls, output_data_list=hybrid_out)
