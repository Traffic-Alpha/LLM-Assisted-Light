'''
@Author: WANG Maonan
@Date: 2023-09-19 19:35:26
@Description: 计算两个列表的欧式距离, 如果两个列表长度不同, 则进行 padding
@LastEditTime: 2023-09-19 19:35:27
'''
import numpy as np

def euclidean_distance(a, b):
    # 获取较长列表的长度
    max_length = max(len(a), len(b))
    
    # 使用0进行填充，使两个列表长度相等
    a_padded = np.pad(a, (0, max_length - len(a)), mode='constant')
    b_padded = np.pad(b, (0, max_length - len(b)), mode='constant')
    
    # 计算欧氏距离
    distance = np.sqrt(np.sum((a_padded - b_padded) ** 2))
    return distance

