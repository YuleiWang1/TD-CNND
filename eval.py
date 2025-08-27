import torch
import time
import numpy as np
from numpy import *
import scipy.io as sio
from torch.utils.data import DataLoader
from tool import Pixel
from dataset import data_generator_eval, results_eval
from cnnd import CNND



def main():
    # 分配到的GPU或CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # 加载数据
    load_fn = './data/Beach.mat'
    load_data = sio.loadmat(load_fn)
    data = load_data['data']
    data = data.astype(np.float32)
    [w, h, bs] = data.shape
    print(data.shape)

    print("不要急,正在生成测试数据........")
    x, num = data_generator_eval(data, 3, 5)
    print("生成完de测试数据的大小是:", np.array(x).shape)

    mydataset = Pixel(x)
    train_dataloader = DataLoader(mydataset, batch_size=256, shuffle=False)

    model = torch.load("./model/adjust_cnnd.pth", map_location=device)

    model = model.to(device)

    pre = []
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for i, img in enumerate(train_dataloader):
            # 加载数据
            img = img.to(device)
            y = model(img.unsqueeze(1))
            pre.append(y.cpu())

    dec = results_eval(pre, num)
    output = np.array(dec)
    output = output.reshape(w, h)
    print(output.shape)

    end_time = time.time()
    print("检测时间是:", (end_time - start_time))

    save_fn = './results/output.mat'
    sio.savemat(save_fn, {'o': output})


if __name__ == '__main__':
    main()
