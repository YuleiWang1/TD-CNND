  import torch
import time
from torch import nn
import scipy.io as sio
from numpy import *
import numpy as np
from torch.utils.data import DataLoader
from cnnd import CNND
from VBMF import VBMF
from tensorly.decomposition import partial_tucker
import tensorly as tl

def tucker_decomposition_conv_layer(layer):

    weight = layer.weight.data
    weights = np.array(weight)
    ranks = estimate_ranks(layer)
    # print(weights.shape)
    [core, [a, b]], _ = partial_tucker(weights, modes=[0, 1], rank=ranks, init='svd')

    first_layer = torch.nn.Conv1d(in_channels=b.shape[0], out_channels=b.shape[1], kernel_size=1,
                                  stride=1, padding=0, dilation=layer.dilation, bias=False)

    core_layer = torch.nn.Conv1d(in_channels=core.shape[1], out_channels=core.shape[0], kernel_size=layer.kernel_size,
                                  stride=layer.stride, padding=layer.padding, dilation=layer.dilation, bias=False)

    last_layer = torch.nn.Conv1d(in_channels=a.shape[1], out_channels=a.shape[0], kernel_size=1,
                                  stride=1, padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(torch.from_numpy(b), 1, 0).unsqueeze(-1)
    last_layer.weight.data = torch.from_numpy(a).unsqueeze(-1)
    core_layer.weight.data = torch.from_numpy(core)

    new_layers = [first_layer, core_layer, last_layer]
    new_layer = nn.Sequential(*new_layers)

    # return new_layer
    return new_layer, first_layer, core_layer, last_layer

def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """
    weights = layer.weight.data
    weights = np.array(weights)
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def model_fool(model, n):
    if n == 2:
        model.conv2 = tucker_decomposition_conv_layer(model.conv2)
    elif n == 3:
        model.conv3 = tucker_decomposition_conv_layer(model.conv3)
    elif n == 4:
        model.conv4 = tucker_decomposition_conv_layer(model.conv4)
    elif n == 5:
        model.conv5 = tucker_decomposition_conv_layer(model.conv5)
    elif n == 6:
        model.conv6 = tucker_decomposition_conv_layer(model.conv6)
    elif n == 7:
        model.conv7 = tucker_decomposition_conv_layer(model.conv7)
    elif n == 8:
        model.conv8 = tucker_decomposition_conv_layer(model.conv8)
    elif n == 9:
        model.conv9 = tucker_decomposition_conv_layer(model.conv9)
    elif n == 10:
        model.conv10 = tucker_decomposition_conv_layer(model.conv10)
    elif n == 11:
        model.conv11 = tucker_decomposition_conv_layer(model.conv11)
    elif n == 12:
        model.conv12 = tucker_decomposition_conv_layer(model.conv12)
    elif n == 13:
        model.conv13 = tucker_decomposition_conv_layer(model.conv13)
    elif n == 14:
        model.conv14 = tucker_decomposition_conv_layer(model.conv14)
    elif n == 15:
        model.conv15 = tucker_decomposition_conv_layer(model.conv15)
    elif n == 16:
        model.conv16 = tucker_decomposition_conv_layer(model.conv16)
    return model

def tucker_model(model, num):
    model.eval()
    model = model.cpu()
    for i in range(len(num)):
        model = model_fool(model, num[i])

    return model

def tucker_time(layer, x):
    _, l1, l2, l3 = tucker_decomposition_conv_layer(layer)

    time1 = time.time()
    y = layer(x)

    time2 = time.time()
    y1 = l1(x)

    time3 = time.time()
    y2 = l2(y1)

    time4 = time.time()
    y3 = l3(y2)

    time5 = time.time()

    term0 = time2 - time1
    term1 = time3 - time2
    term2 = time4 - time3
    term3 = time5 - time4
    return term0, term1, term2, term3, (term1+term2+term3)


if __name__ == "__main__":
    # 分配到的GPU或CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    model = torch.load("./model/cnnd.pth", map_location=device)
    model.eval()
    model = model.cpu()

    x = torch.rand(256, 1, 189)
    x = model.conv1(x)
    x = model.conv2(x)
    x = model.conv3(x)
    x = model.conv4(x)
    x = model.conv5(x)
    x = model.conv6(x)
    x = model.conv7(x)
    x = model.conv8(x)
    x = model.conv9(x)
    x = model.conv10(x)
    x = model.conv11(x)
    x = model.conv12(x)
    x = model.conv13(x)
    x = model.conv14(x)
    x = model.conv15(x)
    # x = model.conv16(x)

    layer = model.conv16

    time0, time1, time2, time3, time4 = tucker_time(layer, x)


    print("没分解的时间:", time0)
    print("第一层:", time1)
    print("第二层:", time2)
    print("第三层:", time3)
    print("一、二、三：", time4)




