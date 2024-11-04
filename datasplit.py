#用于划分数据集为验证集和训练集

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_sequence_data(input_file, sequence_length=5, train_ratio=0.8, random_state=42):
    """
    读取序列数据并按序列单位分割为训练集和验证集
    
    Args:
        input_file: 输入文件路径
        sequence_length: 每个序列的长度，默认5
        train_ratio: 训练集比例，默认0.8
        random_state: 随机种子，默认42
    
    Returns:
        train_file: 训练集文件
        val_file: 验证集文件
    """
    # 读取数据
    data = pd.read_csv(input_file, header=None)
    
    # 确保数据总行数是序列长度的整数倍
    total_rows = len(data)
    if total_rows % sequence_length != 0:
        raise ValueError(f"数据总行数({total_rows})不是序列长度({sequence_length})的整数倍")
    
    # 计算序列数量
    num_sequences = total_rows // sequence_length
    
    # 创建序列索引
    sequence_indices = np.arange(num_sequences)
    
    # 获取每个序列的标签(使用第一行的标签)
    sequence_labels = data[0].values[::sequence_length]
    
    # 分割序列索引
    train_indices, val_indices = train_test_split(
        sequence_indices,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=sequence_labels  # 保持标签比例一致
    )
    
    # 将序列索引转换为行索引
    train_row_indices = []
    for idx in train_indices:
        start_row = idx * sequence_length
        train_row_indices.extend(range(start_row, start_row + sequence_length))
    
    val_row_indices = []
    for idx in val_indices:
        start_row = idx * sequence_length
        val_row_indices.extend(range(start_row, start_row + sequence_length))
    
    # 分割数据
    train_data = data.iloc[train_row_indices]
    val_data = data.iloc[val_row_indices]
    
    # 保存为文件
    train_file = 'train_data.csv'
    val_file = 'val_data.csv'
    
    train_data.to_csv(train_file, index=False, header=False)
    val_data.to_csv(val_file, index=False, header=False)
    
    # 打印数据集信息
    print(f"原始数据集:")
    print(f"- 总行数: {len(data)}")
    print(f"- 序列数: {num_sequences}")
    print(f"- 每个序列长度: {sequence_length}")
    
    print(f"\n训练集:")
    print(f"- 行数: {len(train_data)}")
    print(f"- 序列数: {len(train_indices)}")
    
    print(f"\n验证集:")
    print(f"- 行数: {len(val_data)}")
    print(f"- 序列数: {len(val_indices)}")
    
    print("\n类别分布(按序列计算):")
    print("原始数据集:")
    print(pd.Series(sequence_labels).value_counts())
    print("\n训练集:")
    print(pd.Series(data[0].values[::sequence_length][train_indices]).value_counts())
    print("\n验证集:") 
    print(pd.Series(data[0].values[::sequence_length][val_indices]).value_counts())
    
    # 验证每个文件的完整性
    def verify_sequences(file_path, seq_length):
        data = pd.read_csv(file_path, header=None)
        if len(data) % seq_length != 0:
            raise ValueError(f"{file_path}中的行数不是{seq_length}的整数倍")
        return True
    
    verify_sequences(train_file, sequence_length)
    verify_sequences(val_file, sequence_length)
        
    return train_file, val_file

if __name__ == "__main__":
    # 使用示例
    input_file = "E:/Coding_file/Python/CNN/data/raw/data1/annotations_output.csv"
    train_file, val_file = split_sequence_data(input_file)