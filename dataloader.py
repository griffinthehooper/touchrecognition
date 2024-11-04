import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random

class SkeletonDataset(Dataset):
    def __init__(self, file_path, normalize=True):
        """
        初始化数据集
        Args:
            file_path (str): CSV文件路径
            normalize (bool): 是否对数据进行归一化，默认True
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self.normalize = normalize
        self.data, self.labels = self.load_data_from_csv(file_path)
    
    def normalize_data(self, sample):
        """
        归一化数据
        Args:
            sample (numpy.ndarray): 形状为(5, 21, 2)的数组
        Returns:
            numpy.ndarray: 归一化后的数组
        """
        # 对每一帧的坐标进行归一化
        for i in range(sample.shape[0]):  # 遍历每一帧
            # 找到当前帧的中心点
            center = sample[i].mean(axis=0)
            # 减去中心点，使数据中心化
            sample[i] = sample[i] - center
            # 计算标准差
            std = np.std(sample[i])
            if std > 1e-8:  # 避免除以0
                sample[i] = sample[i] / std
        return sample
    
    def load_data_from_csv(self, file_path):
        """
        从CSV文件加载数据
        Args:
            file_path (str): CSV文件路径
        Returns:
            tuple: (data, labels) 张量对
        """
        df = pd.read_csv(file_path, header=None)
        data = []
        labels = []
        
        # 确保总行数是5的倍数
        total_rows = (len(df) // 5) * 5
        df = df.iloc[:total_rows]
        
        for i in range(0, total_rows, 5):
            label = 1 if 1 in df.iloc[i:i+5, 0].values else 0
            sample = df.iloc[i:i+5, 1:].values  # 获取5行坐标数据，忽略第一列（标签）

            # 将数据reshape为 (5帧, 21关节点, 2坐标)
            sample = sample.reshape(5, 21, 2)
            
            # 如果需要归一化
            if self.normalize:
                sample = self.normalize_data(sample)
            
            # 将维度重新排列为 (2通道, 21关节点, 5帧)
            sample = sample.transpose(2, 1, 0)
            
            data.append(sample)
            labels.append(label)
        
        # 转换为numpy array，然后转换为torch tensor
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        # 转换为torch tensor
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]