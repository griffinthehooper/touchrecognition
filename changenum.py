import pandas as pd

def replace_value_in_first_column(input_file, output_file, old_value, new_value):
    """
    替换 CSV 文件第一列中的特定数值
    
    参数:
    input_file: 输入CSV文件的路径
    output_file: 输出CSV文件的路径
    old_value: 要替换的原始数值
    new_value: 新的数值
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 获取第一列的列名
    first_column = df.columns[0]
    
    # 仅在第一列中替换指定的值
    df[first_column] = df[first_column].replace(old_value, new_value)
    
    # 保存修改后的文件
    df.to_csv(output_file, index=False)
    print(f"文件已保存至: {output_file}")

# 使用示例
if __name__ == "__main__":
    input_file = "E:/ran/Pictures/Camera Roll/data4/output.csv"       # 输入文件名
    output_file = "E:/ran/Pictures/Camera Roll/data4/output3.csv"     # 输出文件名
    old_value = 1                  # 要替换的原始数值
    new_value = 3                 # 新的数值
    
    replace_value_in_first_column(input_file, output_file, old_value, new_value)