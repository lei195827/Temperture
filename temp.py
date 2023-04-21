import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from joblib import dump, load
# 将数据标准化
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 8)
        self.fc2 = nn.Linear(8, 8)
        self.out = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # dropout with probability 0.2

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))  # apply dropout after activation
        x = self.dropout(self.relu(self.fc2(x)))  # apply dropout after activation
        x = self.out(x)
        return x


def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, criterion, epoches):
    # 创建一个空列表来存储每个epoch的测试集loss值
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

            # 前向传播
            output = model(X)
            loss = criterion(output, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 将loss值添加到列表中
            batch_loss_values.append(loss.item())

        # 计算平均loss值，并将其添加到另一个列表中
        avg_loss = sum(batch_loss_values) / len(batch_loss_values)

        # 在每个epoch结束时，测试模型
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for X, y in test_dataloader:
                # 将数据移动到GPU上
                X, y = X.cuda(), y.cuda()

                # 前向传播
                output = model(X)

                # 计算测试集loss值
                test_loss += criterion(output, y).item() * len(X)

            test_loss /= len(test_dataloader.dataset)
            test_loss_values.append(test_loss)
            print(f"Epoch {epoch}, test loss: {test_loss:.4f}")

    # 使用matplotlib库来绘制折线图
    plt.plot(range(epoches), test_loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()


if __name__ == '__main__':
    # 读取文件路径和表名
    train_xls = "train/1.xls"
    test_xls = "train/2.xls"
    train_data = pd.read_excel(train_xls, sheet_name="data")
    test_data = pd.read_excel(test_xls, sheet_name="data")
    train_data = train_data.loc[:, ~train_data.columns.isin([' Brightness', ' Show Time'])]
    test_data = test_data.loc[:, ~test_data.columns.isin([' Brightness', ' Show Time'])]
    print(train_data.columns)
    # print(train_data.head())
    # print(test.head())
    print("train_data and test_data shape", train_data.shape, test_data.shape)
    # print(label)
    train_data = train_data.dropna()
    # 将数据标准化
    scaler = StandardScaler()
    all_feature = train_data.iloc[:, :-1]
    test_feature = test_data.iloc[:, :-1]
    labels = train_data.iloc[:, -1]
    labels = labels.values.reshape(-1, 1)

    test_labels = test_data.iloc[:, -1]
    test_labels = test_labels.values.reshape(-1, 1)

    all_feature = scaler.fit_transform(all_feature)
    # 保存scaler对象到文件
    dump(scaler, 'scaler.joblib')
    test_feature = scaler.transform(test_feature)
    # 初始化神经网络模型和损失函数
    model = MLP().cuda()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    train_dataset = CustomDataset(all_feature, labels)
    test_dataset = CustomDataset(test_feature, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    epoches = 15

    test_loss_values = []
    predit_values = []
    # train_and_evaluate(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
    # optimizer=optimizer, criterion=criterion, epoches=epoches)
    torch.save(model, 'model.pth')
