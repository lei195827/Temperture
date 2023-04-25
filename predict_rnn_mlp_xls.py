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

from temp_rnn_2 import GRU, RNN_2, LSTM, BiGRU


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
    test_xls = "test/8.xls"
    # 加载已经训练好的模型
    model_mlp, device = predict_xls_mlp.load_model_mlp('model_mlp.pth', input_size=input_size)
    model_rnn, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_rnn.pth', input_size=input_size,
                                                       hidden_size=hidden_size,
                                                       num_layers=num_layers, fc_size=fc_size,
                                                       output_size=output_size, net=RNN_2)
    model_gru, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_gru.pth', input_size=input_size,
                                                       hidden_size=hidden_size,
                                                       num_layers=num_layers, fc_size=fc_size,
                                                       output_size=output_size, net=GRU)
    model_lstm, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_lstm.pth', input_size=input_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers, fc_size=fc_size,
                                                        output_size=output_size, net=LSTM)
    input_data_mlp = predict_xls_mlp.load_test_data_mlp(test_xls, scaler, device, column_xls='columns.xlsx')
    input_tensor_rnn, output = predict_rnn_xls.load_test_data_rnn(test_xls, scaler, device, column_xls='columns.xlsx',
                                                                  sequence_length=sequence_length)
    mlp_out = predict_xls_mlp.predict_temperature_mlp(model_mlp, input_data_mlp)
    rnn_out = predict_rnn_xls.predict_temperature_rnn(model_rnn, input_tensor_rnn,
                                                      sequence_length=sequence_length,
                                                      input_size=input_size)
    gru_out = predict_rnn_xls.predict_temperature_rnn(model_gru, input_tensor_rnn,
                                                      sequence_length=sequence_length,
                                                      input_size=input_size)
    lstm_out = predict_rnn_xls.predict_temperature_rnn(model_lstm, input_tensor_rnn,
                                                       sequence_length=sequence_length,
                                                       input_size=input_size)

    # 可视化预测结果和原始值
    plot_length = 6000
    index_rnn = range(sequence_length - 1, min(plot_length, len(rnn_out) + sequence_length - 1))
    index_mlp = range(min(plot_length, len(mlp_out)))
    np_mlp_out_all = np.asarray(mlp_out).reshape(-1)
    np_rnn_out_all = np.asarray(rnn_out).reshape(-1)
    np_gru_out_all = np.asarray(gru_out).reshape(-1)
    np_lstm_out_all = np.asarray(lstm_out).reshape(-1)

    np_mlp_out = np.asarray(mlp_out[:plot_length]).reshape(-1)
    np_rnn_out = np.asarray(rnn_out[:plot_length]).reshape(-1)
    np_gru_out = np.asarray(gru_out[:plot_length]).reshape(-1)
    np_lstm_out = np.asarray(lstm_out[:plot_length]).reshape(-1)

    mlp_ratio = 0.2
    rnn_ratio = 0.2
    gru_ratio = 0.3
    lstm_ratio = 0.3
    hybrid_out = np.zeros(len(np_mlp_out_all))
    hybrid_out[:sequence_length] = np_mlp_out[:sequence_length]
    hybrid_out[sequence_length:] = rnn_ratio * np_rnn_out_all[:] + mlp_ratio * np_mlp_out_all[sequence_length:] \
                                   + gru_ratio * np_gru_out_all[:] + lstm_ratio * np_lstm_out_all[:]
    # 计算每个预测数组和其对应的真实值数组之间的均方根损失
    mlp_rmse = mean_squared_error(output, np_mlp_out_all, squared=False)
    rnn_rmse = mean_squared_error(output[sequence_length:], np_rnn_out_all,
                                  squared=False)
    gru_rmse = mean_squared_error(output[sequence_length:], np_gru_out_all,
                                  squared=False)
    lstm_rmse = mean_squared_error(output[sequence_length:], np_lstm_out_all,
                                   squared=False)
    hybrid_rmse = mean_squared_error(output[:], hybrid_out[:], squared=False)

    # 创建画布和两个子图
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 8))

    ax1.plot(index_rnn, np_rnn_out[:plot_length - (sequence_length - 1)], label='RNN_Temp')
    ax1.plot(index_rnn, np_gru_out[:plot_length - (sequence_length - 1)], label='GRU_Temp')
    ax1.plot(index_rnn, np_lstm_out[:plot_length - (sequence_length - 1)], label='LSTM_Temp')

    if np.any(output):
        ax1.plot(index_mlp, output[:plot_length], label='GT_Temp')
    else:
        print("all of output if '0'")
    ax1.plot(index_mlp, np_mlp_out[:], label='MLP_Temp')
    ax1.plot(index_mlp, hybrid_out[:plot_length], label='Hybrid_Temp')
    ax1.legend()

    # 在第二个子图中绘制表格
    table_data = [['Model', 'RMSE'],
                  ['MLP', f'{mlp_rmse:.3f}'],
                  ['RNN', f'{rnn_rmse:.3f}'],
                  ['GRU', f'{gru_rmse:.3f}'],
                  ['LSTM', f'{lstm_rmse:.3f}'],
                  ['Hybrid', f'{hybrid_rmse:.3f}']]
    ax2.axis('off')
    ax2.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')

    # 调整布局
    plt.subplots_adjust(hspace=0)

    plt.savefig(os.path.join("result", f'{Path(test_xls).stem}_{plot_length}.png'), dpi=300,
                bbox_inches='tight')  # 保存图片为png格式，分辨率为300，裁剪掉多余的空白
    plt.show()

    # 打印均方根损失
    print(f'MLP RMSE:{mlp_rmse:.3f}')
    print(f'RNN RMSE:{rnn_rmse:.3f}')
    print(f'GRU RMSE:{gru_rmse:.3f}')
    print(f'LSTM RMSE:{lstm_rmse:.3f}')
    print(f'Hybrid RMSE:{hybrid_rmse:.3f}')

    add_predict_data(test_xls=test_xls, output_data_list=hybrid_out)
