import pandas as pd
import numpy as np

def process_click_data(input_file, output_file):
    """
    Process click data to create labeled sequences around click events.
    """
    try:
        # 读取数据并正确分割
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # 将每行数据分割成列表
        data = []
        for line in lines:
            # 移除可能的换行符并分割
            values = line.strip().split(',')
            if len(values) > 1:  # 确保有足够的值
                try:
                    # 转换第一列为整数，其余为浮点数
                    row = [int(values[0])] + [float(x) for x in values[1:] if x]
                    data.append(row)
                except ValueError as e:
                    print(f"跳过无效行: {line[:50]}... 错误: {e}")
                    continue
                    
        # 创建DataFrame
        df = pd.DataFrame(data)
        print(f"成功读取数据，共 {len(df)} 行")
        
        # 打印调试信息
        print("\n数据前5行:")
        print(df.head())
        
        print("\n第一列的值类型:")
        print(df[0].dtype)
        
        print("\n第一列的唯一值:")
        print(df[0].unique())
        
        # 找到所有点击事件（第一列为1的帧）
        click_indices = df.index[df[0] == 1].tolist()
        print(f"\n找到 {len(click_indices)} 个点击事件")
        print("点击事件的索引:", click_indices)
        
        if not click_indices:
            print("数据中没有找到点击事件（第一列值为1的行）")
            return
            
        # 初始化结果列表
        processed_windows = []
        
        for click_idx in click_indices:
            # 对于每个点击事件，处理三个窗口
            # 确定窗口范围
            pre_window_start = max(0, click_idx - 7)  # 点击前5帧
            pre_window_end = max(0, click_idx - 2)
            
            click_window_start = max(0, click_idx - 2)  # 点击帧及其前后2帧
            click_window_end = min(len(df), click_idx + 3)
            
            post_window_start = min(len(df), click_idx + 3)  # 点击后5帧
            post_window_end = min(len(df), click_idx + 8)
            
            # 提取并标注窗口
            if pre_window_start < pre_window_end:
                pre_window = df.iloc[pre_window_start:pre_window_end].copy()
                pre_window.iloc[:, 0] = 0  # 标注为负样本
                processed_windows.append(pre_window)
            
            if click_window_start < click_window_end:
                click_window = df.iloc[click_window_start:click_window_end].copy()
                click_window.iloc[:, 0] = 1  # 标注为正样本
                processed_windows.append(click_window)
                
            if post_window_start < post_window_end:
                post_window = df.iloc[post_window_start:post_window_end].copy()
                post_window.iloc[:, 0] = 0  # 标注为负样本
                processed_windows.append(post_window)
        
        # 合并所有处理后的窗口
        if processed_windows:
            processed_data = pd.concat(processed_windows, axis=0)
            # 按原始索引排序
            processed_data = processed_data.sort_index()
            # 移除重复行（保留第一次出现的）
            processed_data = processed_data.loc[~processed_data.index.duplicated(keep='first')]
            
            # 保存处理后的数据
            processed_data.to_csv(output_file, sep=',', header=False, index=False)
            
            # 打印统计信息
            total_frames = len(processed_data)
            positive_frames = len(processed_data[processed_data[0] == 1])
            negative_frames = len(processed_data[processed_data[0] == 0])
            
            print(f"\n处理完成，文件已保存。")
            print(f"总帧数: {total_frames}")
            print(f"正样本帧数: {positive_frames}")
            print(f"负样本帧数: {negative_frames}")
            
        else:
            print("处理后没有有效数据")
    
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # 设置输入输出路径
    input_file = "E:/Coding_file/Python/CNN/data/raw/data1/annotations.csv"  # 请修改为实际输入文件路径
    output_file = "E:/Coding_file/Python/CNN/data/raw/data1/annotations_output.csv"  # 请修改为实际输出文件路径
    
    process_click_data(input_file, output_file)





