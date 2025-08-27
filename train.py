import time
import torch
import math
from torch import nn
import scipy.io as sio
import numpy as np
from numpy import *
import torch.optim.optimizer
from torch.utils.data import DataLoader
from cnnd import CNND
from tool import MyDataset
from dataset import training_data_generator



def main():
    # 分配到的GPU或CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # 加载数据
    load_fn = './data/Salinas.mat'
    load_data = sio.loadmat(load_fn)

    data0 = load_data['data']
    data0 = data0.astype(np.float32)

    labels0 = load_data['map']
    labels0 = labels0.astype(np.float32)

    # 生成训练数据
    data, labels = training_data_generator(data0, labels0, sim_samples=100, dis_samples=600)
    data = torch.tensor(data, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)

    print("训练数据集的长度为：{}".format(len(data)))
    [_, _, bs] = data.shape
    print("data:", data.shape)
    print("labels:", labels.shape)

    # 超参数
    learning_rate = 0.001
    batch_size = 256
    epochs = 50
    criterion = nn.BCELoss()
    # 实例化模型与优化器
    cnnd = CNND(bs=bs).to(device)
    optimizer = torch.optim.Adam(cnnd.parameters(), lr=learning_rate)
    # DataLoader
    mydataset = MyDataset(data, labels)
    data_loader = DataLoader(dataset=mydataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for epoch in range(epochs):
        print("----------第{}轮训练开始了----------".format(epoch + 1))
        cnnd.train()
        total_train_step = 0
        for i, (img, label) in enumerate(data_loader):
            img = img.to(device)
            label = label.to(device)

            outputs = cnnd(img)
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
        if (epoch + 1) % 10 == 0:
            end_time = time.time()
            print("训练时间：{}".format((end_time - start_time)))
            torch.save(cnnd, './model/cnnd_{}.pth'.format((epoch + 1)))
            print("模型已保存")


if __name__ == '__main__':
    main()
