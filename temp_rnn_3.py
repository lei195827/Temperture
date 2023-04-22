import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch import nn
import torch.optim as optim
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR


class CustomDataset_RNN(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.sequence_length

        return self.X[start_index:end_index], self.y[end_index - 1]

    def __len__(self):
        return len(self.X) - self.sequence_length

    def shuffle(self):
        perm = torch.randperm(len(self.X))
        self.X = self.X[perm]
        self.y = self.y[perm]


# 定义神经网络模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class RNN_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fc_size):
        super(RNN_2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fc_size)  # 添加额外的全连接层
        self.fc2 = nn.Linear(fc_size, output_size)  # 输出层
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = F.relu(self.fc1(out))  # 使用ReLU激活函数传递到第一个全连接层
        out = self.fc2(out)  # 传递到输出层
        return out


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fc_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, fc_size)  # 添加额外的全连接层
        self.fc2 = nn.Linear(fc_size, output_size)  # 输出层
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size * 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 初始化双向 RNN 的初始状态
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = F.relu(self.fc1(out))  # 使用ReLU激活函数传递到第一个全连接层
        out = self.fc2(out)  # 传递到输出层
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fc_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fc_size)  # 添加额外的全连接层
        self.fc2 = nn.Linear(fc_size, output_size)  # 输出层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = F.relu(self.fc1(out))  # 使用ReLU激活函数传递到第一个全连接层
        out = self.fc2(out)  # 传递到输出层
        return out


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fc_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, fc_size)  # 添加额外的全连接层，注意输入维度要乘以2
        self.fc2 = nn.Linear(fc_size, output_size)  # 输出层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 注意隐藏状态的维度也要乘以2
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = F.relu(self.fc1(out))  # 使用ReLU激活函数传递到第一个全连接层
        out = self.fc2(out)  # 传递到输出层
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fc_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fc_size)  # 添加额外的全连接层
        self.fc2 = nn.Linear(fc_size, output_size)  # 输出层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = F.relu(self.fc1(out))  # 使用ReLU激活函数传递到第一个全连接层
        out = self.fc2(out)  # 传递到输出层
        return out


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fc_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, fc_size)  # 添加额外的全连接层（注意输出特征数变为 hidden_size*2）
        self.fc2 = nn.Linear(fc_size, output_size)  # 输出层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 注意 h0 需要是双向 LSTM 的形状
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 注意 c0 需要是双向 LSTM 的形状
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = F.relu(self.fc1(out))  # 使用ReLU激活函数传递到第一个全连接层
        out = self.fc2(out)  # 传递到输出层
        return out


def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, criterion, epoches, input_size=8,
                       sequence_length=50):
    # 创建一个空列表来存储每个epoch的测试集loss值
    train_loss_values = []
    test_loss_values = []
    for epoch in range(epoches):
        # 在每个epoch开始时，将训练集中的数据打乱
        train_dataloader.dataset.shuffle()
        # 开始训练
        model.train()
        # 创建一个空列表来存储每个batch的loss值
        batch_loss_values = []
        for i, (X, y) in enumerate(train_dataloader):
            # 将数据移动到GPU上
            X = X.view(-1, sequence_length, input_size)
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            # 前向传播
            output = model(X)
            loss = criterion(output, y)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 将loss值添加到列表中
            batch_loss_values.append(loss.item())
        # 计算平均loss值，并将其添加到另一个列表中
        avg_loss = sum(batch_loss_values) / len(batch_loss_values)
        train_loss_values.append(avg_loss)
        # 在每个epoch结束时，测试模型
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for X, y in test_dataloader:
                # 将数据移动到GPU上
                X = X.view(-1, sequence_length, input_size)
                X, y = X.cuda(), y.cuda()
                output = model(X)
                output = output
                # 计算测试集loss值
                # return F.l1_loss(input, target)
                test_loss += criterion(output, y).item() * len(X)

            test_loss /= len(test_dataloader.dataset)
            test_loss_values.append(test_loss)
            print(f"Epoch {epoch + 1}, test loss: {test_loss:.4f}")

    # 使用matplotlib库来绘制折线图
    plt.plot(range(epoches), train_loss_values, label='train')
    plt.plot(range(epoches), test_loss_values, label='test')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.show()
    return model


def get_number(filename):
    """从文件名中提取数字"""
    return int(os.path.splitext(filename)[0].split("/")[-1])


def process_data(data):
    # 将时间转换为时间戳并转化为秒数
    data[' Show Time'] = pd.to_datetime(data[' Show Time'], format='%Y-%m-%d %H:%M:%S').view('int64') // 10 ** 9
    # 计算相邻时间戳的差值
    data['time_diff'] = data[' Show Time'].diff().fillna(0)

    # 按照每分钟分割数据，并将 time_diff 设为 0
    data.loc[data['time_diff'] > 60, 'time_diff'] = 0
    data['time_diff'] = data['time_diff'].cumsum()

    # 将数据分组并添加每个组的时间差
    data_grouped = data.groupby('time_diff')
    processed_data = []
    for group, group_data in data_grouped:
        group_data = group_data.reset_index(drop=True)
        processed_data.append(group_data)
    return pd.concat(processed_data)


if __name__ == '__main__':
    # 读取文件路径和表名
    train_folder = "train/"
    test_xls = "test/20.xls"
    ori_train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if
                       os.path.isfile(os.path.join(train_folder, f)) and os.path.splitext(f)[1] == ".xls"]
    train_files = sorted(ori_train_files, key=get_number)
    print(train_files)
    # 将所有数据拼接成一个 DataFrame
    train_data = pd.DataFrame()
    NotVal = False
    for file in train_files:
        if os.path.basename(file) != os.path.basename(test_xls) or NotVal:
            df = pd.read_excel(file, sheet_name="data", skiprows=list(range(1, 100)))
            df = df.loc[:, ~df.columns.isin([' Brightness', " fixed humidity", ' fixed temperature'])]
            df = process_data(df)
            train_data = pd.concat([train_data, df])
        else:
            print(f'test xls is:{file}')

    df_test = pd.DataFrame(columns=train_data.columns)
    df_test.to_excel('columns.xlsx', index=False)
    test_data = pd.read_excel(test_xls, sheet_name="data")
    test_data = pd.concat([test_data, df_test])
    test_data = test_data.loc[:, ~test_data.columns.isin([' Brightness', " fixed humidity", ' fixed temperature'])]
    test_data = process_data(test_data)
    train_data = train_data.loc[:,
                 ~train_data.columns.isin([' Brightness', " fixed humidity", ' Show Time', ' fixed temperature'])]
    test_data = test_data.loc[:,
                ~test_data.columns.isin([' Brightness', " fixed humidity", ' Show Time', ' fixed temperature'])]

    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    # print(test_data.head())
    # print(train_data.head())
    print("train_data and test_data shape", train_data.shape, test_data.shape)
    # 将数据标准化
    scaler = StandardScaler()
    scaler.feature_names = ["CPU0_T", "CPU1_T", "CPU2_T", "CPU3_T", "Module_humity", "Module_temp", "lm75",
                            'time_diff', ' IsTalking']
    all_feature = train_data.loc[:, ~train_data.columns.isin(["GT_Temp"])]
    test_feature = test_data.loc[:, ~test_data.columns.isin(["GT_Temp"])]
    print(type(all_feature))
    print(all_feature.shape)

    labels = train_data["GT_Temp"]
    labels = labels.values.reshape(-1, 1)

    test_labels = test_data["GT_Temp"]
    test_labels = test_labels.values.reshape(-1, 1)

    all_feature = scaler.fit_transform(all_feature)
    # 保存scaler对象到文件
    dump(scaler, 'scaler_time.joblib')
    test_feature = scaler.transform(test_feature)

    # 初始化神经网络模型和损失函数
    input_size = 9
    hidden_size = 9
    num_layers = 2
    output_size = 1
    sequence_length = 50
    batch_size = 256
    epoches = 60
    fc_size = 8
    lr = 0.0001
    momentum = 0.9
    weight_decay = 2e-5
    # 创建RNN模型

    model = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size
                , fc_size=fc_size).cuda()
    # criterion = nn.MSELoss()
    # 定义优化器
    criterion = nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=2e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_dataset = CustomDataset_RNN(all_feature, labels, sequence_length=sequence_length)
    test_dataset = CustomDataset_RNN(test_feature, test_labels, sequence_length=sequence_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_loss_values = []
    predit_values = []
    model = train_and_evaluate(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               optimizer=optimizer, input_size=input_size,
                               criterion=criterion, epoches=epoches, sequence_length=sequence_length)
    torch.save(model.state_dict(), "model_gru_time.pth")
    print(f'save model success')
