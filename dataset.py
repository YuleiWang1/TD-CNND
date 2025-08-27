import torch
import numpy as np
from numpy import *


def training_data_generator(data, gt, sim_samples=100, dis_samples=600):
    img = data
    gt = gt
    img = img.astype(np.float32)
    gt = gt.astype(np.float32)

    d = img.shape[2]
    img = img.reshape(-1, d)
    gt = gt.flatten()
    class_num = len(np.unique(gt)) - 1  # except anomaly pixel

    training_data = np.zeros((1, d))
    training_labels = np.zeros(1)

    # dissimilar pixel pair
    for i in range(class_num - 1):
        for j in range(i + 1, class_num):
            index_i = np.where(gt == i + 1)
            index_j = np.where(gt == j + 1)
            data_i = img[index_i][:dis_samples]
            data_j = img[index_j][:dis_samples]

            training_data = np.concatenate((training_data, np.abs(data_i - data_j)), axis=0)

    training_labels = np.concatenate((training_labels, np.ones(int(dis_samples * class_num * (class_num - 1) / 2))))

    # similar pixel pair
    for i in range(class_num):
        data_similar = np.zeros((sim_samples, sim_samples, d))
        index = np.where(gt == i + 1)  # map=1
        data = img[index][:sim_samples]
        for j in range(sim_samples):
            data_similar[j] = np.abs(data - data[j, None])
        data_similar = data_similar[np.triu_indices(sim_samples, k=1)]
        training_data = np.concatenate((training_data, data_similar), axis=0)

    training_labels = np.concatenate((training_labels, np.zeros(int(class_num * sim_samples * (sim_samples - 1) / 2))))
    training_data = np.delete(training_data, 0, 0)
    training_labels = np.delete(training_labels, 0).astype(np.float32)
    training_data = training_data.reshape(training_data.shape[0], 1, training_data.shape[1]).astype(np.float32)
    return training_data, training_labels

def data_generator_eval(img, inner, outer):
    [w, h, _] = img.shape
    pre = data_generator_mask(img, inner, outer)
    yy = []
    num = []
    for i in range(w*h):
        n, _ = pre[i].shape
        num.append(n)
        for j in range(n):
            term = np.array(pre[i][j])
            yy.append(term)

    yy = np.array(yy)
    yy = torch.tensor(yy, dtype=torch.float)
    # print(yy.shape)
    return yy, num

def data_generator_mask(img, inner, outer):
    new_data = []
    mask = bbox_mask_generator(inner, outer)

    radius = int((outer-1)/2)

    img = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), constant_values=-1)

    [w, h, _] = img.shape
    # for i in range(radius, w-radius):
    #     for j in range(radius, h-radius):
    for i in range(radius, w-radius):
        for j in range(radius, h-radius):
            pixel = img[i, j, None]
            bbox = img[i-radius:i+radius+1, j-radius:j+radius+1]
            bbox = bbox[mask]
            bbox = bbox[bbox[:, 0] > 0]
            term = np.abs(bbox - pixel)
            term = torch.tensor(term, dtype=torch.float)
            new_data.append(term)

    return new_data

def bbox_mask_generator(inner, outer):

    mask = np.ones([outer]*2)
    zero_index = int((outer-inner)/2)
    mask[zero_index:-zero_index, zero_index:-zero_index] = 0
    mask = mask.astype(bool)
    return mask

def results_eval(img, num):
    yyy = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            term = img[i][j]
            yyy.append(term)
    yyy = torch.stack(yyy)
    # print(pre.shape)

    dec = []
    sum = 0
    for i in range(len(num)):
        pre = []
        for j in range(num[i]):
            pre.append(yyy[(sum + j)])
        pre = torch.stack(pre)
        pre = torch.mean(pre).item()
        dec.append(pre)
        sum = sum + num[i]

    return dec

