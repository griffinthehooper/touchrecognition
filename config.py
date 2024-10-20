import os
import pandas as pd

# 使用环境变量或配置文件来管理路径
train_file_path = os.getenv('TRAIN_FILE_PATH', './data/window_data3.csv')

def get_features():
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Training file not found: {train_file_path}")
    sample_df = pd.read_csv(train_file_path, header=None, nrows=5)
    features = sample_df.shape[1] - 1  # 去除标签列
    return features
