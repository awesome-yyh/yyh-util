# coding=UTF-8
'''
Author: yangyahe
LastEditors: yangyahe
Date: 2022-08-23 16:52:53
LastEditTime: 2022-08-23 18:04:56
Description: dpp多样性核心代码, 包括核矩阵的构造和滑动窗口式dpp计算, 使用numpy进行矩阵运算
'''
import numpy as np
import time

def dpp_sw(ids, kernel_matrix, window_size, epsilon=1E-10):
    """
    求解dpp的贪心算法的滑动窗口版
    Args:
        ids: 需要打散的文章id list
        kernel_matrix: 2d-np.array, 核矩阵
        window_size: 正整数, 滑动窗口的大小
        epsilon: 一个极小的正数, 终止条件是直到最大边际收益为负时终止
    Returns:
        打散后的文章id列表
    """
    item_size = kernel_matrix.shape[0]
    max_length = item_size
    
    v = np.zeros((max_length, max_length))
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(ids[selected_item])
    window_left_index = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[window_left_index:k, selected_item]
        di_optimal = np.sqrt(di2s[selected_item])
        v[k, window_left_index:k] = ci_optimal
        v[k, k] = di_optimal
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[window_left_index:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        if len(selected_items) >= window_size:
            window_left_index += 1
            for ind in range(window_left_index, k + 1):
                t = np.sqrt(v[ind, ind] ** 2 + v[ind, window_left_index - 1] ** 2)
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
            di2s += np.square(cis[window_left_index - 1, :])
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(ids[selected_item])
    return selected_items

def build_kernel_matrix(scores, feature_vectors, theta = 0.7):
    """
    构建核矩阵
    Args:
        scores: list, 排序分
        feature_vectors: 2d-list, 特征向量
        theta: 0-1间的浮点数, 相关性权重, theta越大相关性越重要, theta越小多样性越重要
    Returns:
        kernel_matrix: 2d-np.array 核矩阵
    """
    item_size = len(scores)
    scores = np.array(scores, dtype=np.float32)
    feature_vectors = np.array(feature_vectors, dtype=np.float32)
    
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    similarities = np.dot(feature_vectors, feature_vectors.T) # 对角线上的元素是scores的平方
    
    alpha = theta / (2.0 * (1-theta)) # 相关性和多样性的trade off
    scores = np.exp(alpha * scores)
    
    similarities = (1 + similarities) / 2 # 需要保证任意两个商品的相似度在0到1之间，而inner product的范围在[-1,1]
    
    kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))
    return kernel_matrix

if __name__ == "__main__":
    item_size = 60
    feature_dimension = 128
    window_size = 5

    scores = np.random.rand(60)*3 # 排序分
    scores = scores.tolist()
    scores.sort(reverse=True)
    # scores = [3,2,1]
    ids = scores
    
    feature_vectors = np.random.randn(item_size, feature_dimension)
    item_vecs_list = feature_vectors.tolist()
    # item_vecs_list = [[4.1, 7.1, 9.1], [4.1, 7.2, 9.2],[1.1, 2.1, 3.2]] # 简易测试数据，前2组非常相似，期望能打散
    
    t1 = time.time()
    kernel_matrix = build_kernel_matrix(scores, item_vecs_list)
    dpp_ids = dpp_sw(ids, kernel_matrix, window_size)
    print(f"dpp_sw: {dpp_ids}")
    print(f"run dpp_sw time: {(time.time() - t1)*1000} ms, len: {len(dpp_ids)}")
