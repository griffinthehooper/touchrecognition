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
        # 添加数据验证
        unique_labels = torch.unique(self.labels)
        print("Unique labels in dataset:", unique_labels)
        if torch.any(self.labels < 0) or torch.any(self.labels >= 4):
            raise ValueError(f"Invalid label values found. Labels should be in range [0,3], got {unique_labels}")
    
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
        # 明确指定数据类型为float
        df = pd.read_csv(file_path, header=None, dtype={0: np.int64})
        
        # 验证标签列的值并进行映射
        original_labels = df[0].unique()
        print("Original unique labels in dataset:", original_labels)
        
        # 映射标签为0,1,2,3
        label_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        df[0] = df[0].map(lambda x: label_mapping.get(x, x))
        
        data = []
        labels = []

        # 确保总行数是5的倍数
        total_rows = (len(df) // 5) * 5
        df = df.iloc[:total_rows]

        for i in range(0, total_rows, 5):
            # 获取当前5帧的所有标签
            frame_labels = df.iloc[i:i+5, 0].values
            
            # 统计当前5帧中各标签的出现次数
            unique, counts = np.unique(frame_labels, return_counts=True)
            label_counts = dict(zip(unique, counts))
            
            # 确定最终标签（如果有非0标签，选择出现次数最多的；如果出现次数相同，选择较大的值）
            label = 0  # 默认为无点击
            max_count = 0
            for l, count in label_counts.items():
                if l != 0 and (count > max_count or (count == max_count and l > label)):
                    label = int(l)
                    max_count = count

            # 获取坐标数据
            sample = df.iloc[i:i+5, 1:].values.astype(np.float32)
            
            # 重塑数据 (5帧, 21关节点, 2坐标)
            try:
                sample = sample.reshape(5, 21, 2)
            except ValueError as e:
                print(f"Error reshaping data at index {i}. Data shape: {sample.shape}")
                continue

            # 归一化
            if self.normalize:
                sample = self.normalize_data(sample)

            # 重排维度 (2通道, 21关节点, 5帧)
            sample = sample.transpose(2, 1, 0)

            data.append(sample)
            labels.append(label)

        if not data:
            raise ValueError("No valid data was loaded")

        # 转换为numpy array
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        print("Final unique labels:", np.unique(labels))
        print("Label distribution:", np.bincount(labels))
        
        # 确保标签在正确范围内
        if np.any(labels < 0) or np.any(labels >= 4):
            raise ValueError(f"Invalid label values found after processing. Labels should be in range [0,3], got {np.unique(labels)}")

        # 转换为torch tensor
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]