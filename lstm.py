# coding: utf-8
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
from sklearn.preprocessing import MinMaxScaler

def split_windows(data, size):
    X = []
    Y = []
    # X作为数据，Y作为标签
    # 滑动窗口，步长为1，构造窗口化数据，每一个窗口的数据标签是窗口末端的GDP值
    for i in range(len(data) - size):
        X.append(data[i:i+size, :])
        Y.append(data[i+size, 1])  # GDP在第二列
    return np.array(X), np.array(Y)

# 加载 CSV 文件
df = pd.read_csv('./data.csv', usecols=['Year', 'GDP'])
all_data = df.values

# 定义训练集和测试集的长度
train_len = int(0.8 * len(all_data))  # 80% 用于训练
train_data = all_data[:train_len, :]
test_data = all_data[train_len:, :]

# 数据可视化
plt.figure(figsize=(12, 8))
plt.plot(train_data[:, 0], train_data[:, 1], label='train data')
plt.plot(test_data[:, 0], test_data[:, 1], label='test data')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.show()

# 归一化处理
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
# 使用训练集的最值对测试集归一化，保证训练集和测试集的分布一致性
scaled_test_data = scaler.transform(test_data)

# 训练集测试集划分
window_size = 7
train_X, train_Y = split_windows(scaled_train_data, size=window_size)
test_X, test_Y = split_windows(scaled_test_data, size=window_size)
print('train shape', train_X.shape, train_Y.shape)
print('test shape', test_X.shape, test_Y.shape)

# 定义模型
class CNN_LSTM(nn.Layer):
    def __init__(self, window_size, fea_num):
        super().__init__()
        self.window_size = window_size
        self.fea_num = fea_num
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=64, stride=1, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=1, padding='same')
        self.dropout = nn.Dropout2D(0.3)

        self.lstm1 = nn.LSTM(input_size=64*fea_num, hidden_size=128, num_layers=1, time_major=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, time_major=False)
        self.fc = nn.Linear(in_features=64, out_features=32)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = x.reshape([x.shape[0], 1, self.window_size, self.fea_num])
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.reshape([x.shape[0], self.window_size, -1])
        x, (h, c) = self.lstm1(x)
        x, (h, c) = self.lstm2(x)
        x = x[:, -1, :]  # 最后一个LSTM只要窗口中最后一个特征的输出
        x = self.fc(x)
        x = self.relu2(x)
        x = self.head(x)

        return x

model = CNN_LSTM(window_size, fea_num=2)
paddle.summary(model, (99, 7, 2))

# 定义超参数
base_lr = 0.005
BATCH_SIZE = 32
EPOCH = 200
lr_schedual = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=EPOCH, verbose=True)
loss_fn = nn.MSELoss()
metric = paddle.metric.Accuracy()
opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr_schedual, beta1=0.9, beta2=0.999)

def process(data, bs):
    l = len(data)
    tmp = []
    for i in range(0, l, bs):
        if i + bs > l:
            tmp.append(data[i:].tolist())
        else:
            tmp.append(data[i:i+bs].tolist())
    tmp = np.array(tmp)
    return tmp

# 处理数据集
train_X = process(train_X, 32)
train_Y = process(train_Y, 32)
print(train_X.shape, train_Y.shape)

# 模型训练
for epoch in range(EPOCH):
    model.train()
    loss_train = 0
    for batch_id, data in enumerate(train_X):
        label = train_Y[batch_id]
        data = paddle.to_tensor(data, dtype='float32')
        label = paddle.to_tensor(label, dtype='float32')
        label = label.reshape([label.shape[0], 1])
        y = model(data)

        loss = loss_fn(y, label)
        opt.clear_grad()
        loss.backward()
        opt.step()
        loss_train += loss.item()
    print("[TRAIN] ========epoch : {},  loss: {:.4f}==========".format(epoch + 1, loss_train))
    lr_schedual.step()

    loss_eval = 0
    model.eval()
    for batch_id, data in enumerate(test_X):
        label = test_Y[batch_id]
        data = paddle.to_tensor(data, dtype='float32')
        label = paddle.to_tensor(label, dtype='float32')
        label = label.reshape([label.shape[0], 1])
        y = model(data)

        loss = loss_fn(y, label)
        loss_eval += loss.item()
    print("[EVAL] ========epoch : {},  loss: {:.4f}==========\n".format(epoch + 1, loss_eval))

# 保存模型参数
paddle.save(model.state_dict(), './work/end2end.params')
paddle.save(lr_schedual.state_dict(), './work/end2end.pdopts')

# 加载模型
model = CNN_LSTM(window_size, fea_num=2)
model_dict = paddle.load('work/end2end.params')
model.load_dict(model_dict)

# 预测未来20年的数据
future_years = 20
future_data = []
last_window = test_X[-1]  # 使用最后一个测试窗口作为初始窗口

for _ in range(future_years):
    last_window = paddle.to_tensor(last_window, dtype='float32').unsqueeze(0)
    prediction = model(last_window)
    future_data.append(prediction.item())
    last_window = np.concatenate((last_window[0][1:, :], np.array([[last_window[0][-1, 0] + 1, prediction.item()]])), axis=0)

# 反归一化
future_data = np.array(future_data).reshape(-1, 1)
future_data = future_data * (scaler.data_max_[1] - scaler.data_min_[1]) + scaler.data_min_[1]

# 生成未来年份
future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 1 + future_years)

# 画图
plt.figure(figsize=(12, 8))
plt.plot(df['Year'], df['GDP'], label='historical data')
plt.plot(future_years, future_data, label='future prediction', marker='*')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
print('RMSE', np.sqrt(mean_squared_error(scaled_test_data[:, 1], test_Y)))
