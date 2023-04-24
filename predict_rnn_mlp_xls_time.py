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

from temp_rnn_3 import GRU, RNN_2, LSTM, BiGRU, BiRNN, BiLSTM


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
    target_seq = 25

    num_layers = 2
    output_size = 1
    test_xls = "train/25.xls"
    unstable = True
    # 加载已经训练好的模型
    model_mlp, device = predict_xls_mlp.load_model_mlp('model_mlp.pth', input_size=input_size)
    model_rnn, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_birnn_all.pth', input_size=input_size,
                                                       hidden_size=hidden_size,
                                                       num_layers=num_layers, fc_size=fc_size,
                                                       output_size=output_size, net=BiRNN)
    model_gru, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_bigru_all.pth', input_size=input_size,
                                                       hidden_size=hidden_size,
                                                       num_layers=num_layers, fc_size=fc_size,
                                                       output_size=output_size, net=BiGRU)
    model_lstm, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_bistm_all.pth', input_size=input_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers, fc_size=fc_size,
                                                        output_size=output_size, net=BiLSTM)

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
    if unstable:
        scaler = load('scaler_unstable.joblib')
        input_tensor_rnn_unstable, output = predict_rnn_xls.load_test_data_rnn(test_xls, scaler, device,
                                                                      column_xls='columns.xlsx',
                                                                      sequence_length=sequence_length)
        model_rnn_unstable, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_birnn_unstable.pth', input_size=input_size,
                                                           hidden_size=hidden_size,
                                                           num_layers=num_layers, fc_size=fc_size,
                                                           output_size=output_size, net=BiRNN)
        model_gru_unstable, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_bigru_unstable.pth', input_size=input_size,
                                                           hidden_size=hidden_size,
                                                           num_layers=num_layers, fc_size=fc_size,
                                                           output_size=output_size, net=BiGRU)
        model_lstm_unstable, device = predict_rnn_xls.load_model_rnn(model_rnn_path='model_bilstm_unstable.pth', input_size=input_size,
                                                            hidden_size=hidden_size,
                                                            num_layers=num_layers, fc_size=fc_size,
                                                            output_size=output_size, net=BiLSTM)
        rnn_out_unstable = predict_rnn_xls.predict_temperature_rnn(model_rnn_unstable, input_tensor_rnn_unstable,
                                                          sequence_length=sequence_length,
                                                          input_size=input_size)
        gru_out_unstable = predict_rnn_xls.predict_temperature_rnn(model_gru_unstable, input_tensor_rnn_unstable,
                                                          sequence_length=sequence_length,
                                                          input_size=input_size)
        lstm_out_unstable = predict_rnn_xls.predict_temperature_rnn(model_lstm_unstable, input_tensor_rnn_unstable,
                                                           sequence_length=sequence_length,
                                                           input_size=input_size)

    # 可视化预测结果和原始值
    print(f'output.var():{output.var()}')
    plot_length = 10000
    if unstable:
        index_rnn_unstable = range(target_seq - 1, min(plot_length, len(rnn_out) + target_seq - 1))
        index_mlp_unstable = range(min(plot_length, len(mlp_out)))
        np_rnn_out_unstable = np.asarray(rnn_out_unstable).reshape(-1)
        np_gru_out_unstable = np.asarray(gru_out_unstable).reshape(-1)
        np_lstm_out_unstable = np.asarray(lstm_out_unstable).reshape(-1)

        np_rnn_out_unstable = np.asarray(rnn_out[:plot_length]).reshape(-1)
        np_gru_out_unstable = np.asarray(gru_out[:plot_length]).reshape(-1)
        np_lstm_out_unstable = np.asarray(lstm_out[:plot_length]).reshape(-1)

        np_rnn_out_unstable = np.asarray(rnn_out_unstable[:plot_length]).reshape(-1)
        np_gru_out_unstable = np.asarray(gru_out_unstable[:plot_length]).reshape(-1)
        np_lstm_out_unstable = np.asarray(lstm_out_unstable[:plot_length]).reshape(-1)

    index_rnn = range(target_seq - 1, min(plot_length, len(rnn_out) + target_seq - 1))
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
    hybrid_out[:target_seq] = np_mlp_out[:target_seq]
    hybrid_out[-target_seq:] = np_mlp_out[-target_seq:]
    hybrid_out[target_seq:-target_seq] = rnn_ratio * np_rnn_out_all[:] + mlp_ratio * np_mlp_out_all[target_seq:-target_seq] \
                                   + gru_ratio * np_gru_out_all[:] + lstm_ratio * np_lstm_out_all[:]

    # 计算每个预测数组和其对应的真实值数组之间的均方根损失
    mlp_rmse = mean_squared_error(output, np_mlp_out_all, squared=False)
    rnn_rmse = mean_squared_error(output[target_seq:-target_seq], np_rnn_out_all,
                                  squared=False)
    gru_rmse = mean_squared_error(output[target_seq:-target_seq], np_gru_out_all,
                                  squared=False)
    lstm_rmse = mean_squared_error(output[target_seq:-target_seq], np_lstm_out_all,
                                   squared=False)
    hybrid_rmse = mean_squared_error(output[:], hybrid_out[:], squared=False)
    if unstable:
        rnn_ratio_unstable = 0.4
        gru_ratio_unstable = 0.3
        lstm_ratio_unstable = 0.3
        hybrid_out_unstable = np.zeros(len(np_mlp_out_all))
        hybrid_out_unstable[:target_seq] = np_mlp_out[:target_seq]
        hybrid_out_unstable[-target_seq:] = np_mlp_out[-target_seq:]
        hybrid_out_unstable[target_seq:-target_seq] = rnn_ratio_unstable * np_rnn_out_unstable[:] + \
        gru_ratio_unstable * np_gru_out_unstable[:] + lstm_ratio_unstable * np_lstm_out_unstable[:]
        hybrid_out_unstable = 0.2*hybrid_out+0.8*hybrid_out_unstable
        rnn_rmse_unstable = mean_squared_error(output[target_seq:-target_seq], np_rnn_out_unstable,
                                      squared=False)
        gru_rmse_unstable = mean_squared_error(output[target_seq:-target_seq], np_gru_out_unstable,
                                      squared=False)
        lstm_rmse_unstable = mean_squared_error(output[target_seq:-target_seq], np_lstm_out_unstable,
                                       squared=False)
        hybrid_rmse_unstable = mean_squared_error(output[:], hybrid_out_unstable[:], squared=False)
    # 创建画布和两个子图
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 8))

    if unstable:
        ax1.plot(index_rnn, (np_rnn_out*0.2+np_rnn_out_unstable*0.8)[:plot_length - (target_seq - 1)], label='RNN_Temp')
        ax1.plot(index_rnn, (np_gru_out*0.2+np_gru_out_unstable*0.8)[:plot_length - (target_seq - 1)], label='GRU_Temp')
        ax1.plot(index_rnn, (np_lstm_out*0.2+np_lstm_out_unstable*0.8)[:plot_length - (target_seq - 1)], label='LSTM_Temp')
        ax1.plot(index_mlp, hybrid_out_unstable[:plot_length], label='Hybrid_Temp')
    else:
        ax1.plot(index_rnn, np_rnn_out[:plot_length - (target_seq - 1)], label='RNN_Temp')
        ax1.plot(index_rnn, np_gru_out[:plot_length - (target_seq - 1)], label='GRU_Temp')
        ax1.plot(index_rnn, np_lstm_out[:plot_length - (target_seq - 1)], label='LSTM_Temp')
        ax1.plot(index_mlp, hybrid_out[:plot_length], label='Hybrid_Temp')


    if np.any(output):
        ax1.plot(index_mlp, output[:plot_length], label='GT_Temp')
    else:
        print("all of output if '0'")
    ax1.plot(index_mlp, np_mlp_out[:], label='MLP_Temp')

    ax1.legend()

    # 在第二个子图中绘制表格
    if unstable:
        table_data = [['Model', 'RMSE'],
                      ['MLP', f'{mlp_rmse:.3f}'],
                      ['RNN', f'{rnn_rmse:.3f}'],
                      ['GRU', f'{gru_rmse:.3f}'],
                      ['LSTM', f'{lstm_rmse:.3f}'],
                      ['Hybrid', f'{hybrid_rmse:.3f}'],
                      ['RNN_u', f'{rnn_rmse_unstable:.3f}'],
                      ['GRU_u', f'{gru_rmse_unstable:.3f}'],
                      ['LSTM_u', f'{lstm_rmse_unstable:.3f}'],
                      ['Hybrid_u', f'{hybrid_rmse_unstable:.3f}']
                      ]
    else:
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
    if unstable:
        add_predict_data(test_xls=test_xls, output_data_list=hybrid_out_unstable)
    else:
        add_predict_data(test_xls=test_xls, output_data_list=hybrid_out)