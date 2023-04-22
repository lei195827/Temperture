import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from joblib import dump, load
# 将数据标准化
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def shuffle(self):
        perm = torch.randperm(len(self.X))
        self.X = self.X[perm]
        self.y = self.y[perm]


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


def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, criterion, epoches):
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
                X, y = X.cuda(), y.cuda()
                output = model(X)
                # 计算测试集loss值
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
    for file in train_files :
        if os.path.basename(file) != os.path.basename(test_xls) or True:
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
    print(test_data.head())
    print(train_data.head())
    print("train_data and test_data shape", train_data.shape, test_data.shape)

    # 将数据标准化
    print(train_data.head())
    print(train_data.head(-5))
    scaler = StandardScaler()
    scaler.feature_names = ["CPU0_T", "CPU1_T", "CPU2_T", "CPU3_T" "Module_humity", "Module_temp", "lm75",
                            ' IsTalking']
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
    dump(scaler, 'scaler.joblib')
    test_feature = scaler.transform(test_feature)
    # 初始化神经网络模型和损失函数
    model = MLP(input_size=8).cuda()
    # criterion = nn.MSELoss()
    # 定义优化器
    criterion = nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=2e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=2e-4)

    train_dataset = CustomDataset(all_feature, labels)
    test_dataset = CustomDataset(test_feature, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    epoches = 40
    test_loss_values = []
    predit_values = []
    model = train_and_evaluate(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               optimizer=optimizer,
                               criterion=criterion, epoches=epoches)
    torch.save(model.state_dict(), "model_mlp.pth")
