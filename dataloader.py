import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
import random

class SkeletonDataset(Dataset):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self.data, self.labels = self.load_data_from_csv(file_path)
    
    def load_data_from_csv(self, file_path):
        df = pd.read_csv(file_path, header=None)
        data = []
        labels = []
        
        total_rows = (len(df) // 5) * 5
        df = df.iloc[:total_rows]
        
        for i in range(0, total_rows, 5):
            label = 1 if 1 in df.iloc[i:i+5, 0].values else 0
            sample = df.iloc[i:i+5, 1:].values
            sample = sample.reshape(5, 21, 2)
            sample = sample.transpose(2, 1, 0)
            
            data.append(sample)
            labels.append(label)
        
        data = np.array(data)
        labels = np.array(labels)
        
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).float()

        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        
        # Simple data augmentation: randomly flip x coordinates
        if random.random() > 0.5:
            data[0, :, :] = 1 - data[0, :, :]  # Assuming x coordinates are normalized to [0,1]
        
        return data, label

def get_sampler(dataset):
    labels = [int(label.item()) for label in dataset.labels]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler, class_counts