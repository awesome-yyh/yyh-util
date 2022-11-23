# coding=UTF-8
'''
Author: yangyahe
LastEditors: yangyahe
Date: 2022-08-23 16:52:53
LastEditTime: 2022-08-23 18:04:56
Description: flask服务, 输入文章id或id列表, 从Redis中读取并返回对应的文章向量, 向量由召回模型训练所得
'''
import numpy as np
import torch
import torch.nn.functional as F
import time
import requests,json
import logging


# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # linux
device=torch.device("mps") # mac m1 gpu: mps

def dpp_sw(ids, kernel_matrix, window_size, epsilon=1E-10):
    """
    Sliding window version of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param window_size: positive int
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    kernel_matrix = kernel_matrix
    item_size = kernel_matrix.shape[0]
    max_length = item_size
    
    v=torch.zeros([max_length,max_length],dtype=torch.float32).to(device)
    cis=torch.zeros([max_length,item_size],dtype=torch.float32).to(device)
    di2s = torch.diag(kernel_matrix)
    selected_items = list()
    selected_item = torch.argmax(di2s)
    selected_items.append(ids[selected_item])
    window_left_index = 0
    while len(selected_items) < max_length:
        k = len(selected_items) #- 1
        ci_optimal = cis[window_left_index:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        v[k, window_left_index:k] = ci_optimal
        v[k, k] = di_optimal
        elements = kernel_matrix[selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[window_left_index:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= torch.square(eis)
        if len(selected_items) >= window_size:
            window_left_index += 1
            for ind in range(window_left_index, k + 1):
                t = torch.sqrt(v[ind, ind] ** 2 + v[ind, window_left_index - 1] ** 2)
                c = t / v[ind, ind]
                s = v[ind, window_left_index - 1] / v[ind, ind]
                v[ind, ind] = t
                v[ind + 1:k + 1, ind] += s * v[ind + 1:k + 1, window_left_index - 1]
                v[ind + 1:k + 1, ind] /= c
                v[ind + 1:k + 1, window_left_index - 1] *= c
                v[ind + 1:k + 1, window_left_index - 1] -= s * v[ind + 1:k + 1, ind]
                cis[ind, :] += s * cis[window_left_index - 1, :]
                cis[ind, :] /= c
                cis[window_left_index - 1, :] *= c
                cis[window_left_index - 1, :] -= s * cis[ind, :]
            di2s += torch.square(cis[window_left_index - 1, :])
        di2s[selected_item] = -torch.inf
        selected_item = torch.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(ids[selected_item])
    return selected_items

def build_kernel_matrix(scores, feature_vectors, theta = 0.7):
    # scores: list, 排序分
    # feature_vectors: 2d-list, 特征向量
    # return: kernel_matrix: 2d-np.array 核矩阵
    # theta越大相关性越重要，theta越小多样性越重要
    
    item_size = len(scores)
    scores = torch.tensor(scores, dtype=torch.float32).to(device)
    feature_vectors = torch.tensor(feature_vectors, dtype=torch.float32).to(device)
    
    feature_vectors = F.normalize(feature_vectors, p=2, dim=1) # dim=1按行处理, l2范数归一化再点乘 = 余弦相似度
    similarities = torch.matmul(feature_vectors, feature_vectors.T) # 对角线上的元素是scores的平方
    
    alpha = theta / (2.0 * (1-theta)) # 相关性和多样性的trade off
    scores = torch.exp(alpha * scores)
    
    similarities = (1 + similarities) / 2 # 需要保证任意两个商品的相似度在0到1之间，而inner product的范围在[-1,1]
    
    kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))
    return kernel_matrix

if __name__ == "__main__":
    item_size = 60
    feature_dimension = 128
    window_size = 5
    np.random.seed(0)
    
    scores = np.random.rand(item_size)*3 # 排序分
    scores = scores.tolist()
    scores.sort(reverse=True)
    ids = scores
    
    feature_vectors = np.random.randn(item_size, feature_dimension)
    # feature_vectors = np.array([[1.1, 2.1, 3.2], [4.1, 7.1, 9.1], [4.1, 7.2, 9.2]])
    item_vecs_list = feature_vectors.tolist()
    
    t1 = time.time()
    kernel_matrix = build_kernel_matrix(scores, item_vecs_list)
    # print(kernel_matrix)
    dpp_ids = dpp_sw(ids, kernel_matrix, window_size)
    print(f"dpp_sw: {dpp_ids}")
    print(f"run dpp_sw time: {(time.time() - t1)*1000} ms, len: {len(dpp_ids)}")
