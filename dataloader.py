import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self.data, self.labels = self.load_data_from_csv(file_path)
    
    def load_data_from_csv(self, file_path):
        df = pd.read_csv(file_path, header=None)
        data = []
        labels = []
        
        # 确保总行数是5的倍数
        total_rows = (len(df) // 5) * 5
        df = df.iloc[:total_rows]
        
        for i in range(0, total_rows, 5):
            label = 1 if 1 in df.iloc[i:i+5, 0].values else 0
            sample = df.iloc[i:i+5, 1:].values  # 获取5行坐标数据，忽略第一列（标签）

            # 将数据reshape为 (5帧, 21关节点, 2坐标)，其中最后一维是x和y坐标
            sample = sample.reshape(5, 21, 2)
            
            # 将维度重新排列为 (2通道, 21关节点, 5帧)
            sample = sample.transpose(2, 1, 0)
            
            data.append(sample)
            labels.append(label)
        
        # 转换为numpy array，然后转换为torch tensor
        data = np.array(data)
        labels = np.array(labels)
        
        # 转换为torch tensor，并确保类型正确
        data = torch.from_numpy(data).float()  # (num_samples, 2, 21, 5)
        labels = torch.from_numpy(labels).float()  # 标签

        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]