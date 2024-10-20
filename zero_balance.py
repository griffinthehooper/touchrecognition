import pandas as pd
import numpy as np
import random

# 读取CSV文件
df = pd.read_csv('E:/Coding_file/Python/CNN/data/window_data3.csv', header=None)

# 按5行分割数据
units = [df.iloc[i:i+5] for i in range(0, len(df), 5)]

# 统计0和1的单位数目
count_0 = sum([1 for unit in units if unit.iloc[:, 0].sum() == 0])
count_1 = sum([1 for unit in units if unit.iloc[:, 0].sum() == 5])

# 打印统计信息
print(f"0的单位数目: {count_0}")
print(f"1的单位数目: {count_1}")

# 如果0或者1的数目过高，随机删除数目更多的单位
if count_0 > count_1:
    units_to_keep = [unit for unit in units if unit.iloc[:, 0].sum() == 5] + \
                    random.sample([unit for unit in units if unit.iloc[:, 0].sum() == 0], count_1)
elif count_1 > count_0:
    units_to_keep = [unit for unit in units if unit.iloc[:, 0].sum() == 0] + \
                    random.sample([unit for unit in units if unit.iloc[:, 0].sum() == 5], count_0)
else:
    units_to_keep = units

# 合并保留的单位并输出为新的CSV文件
result = pd.concat(units_to_keep)
result.to_csv('E:/Coding_file/Python/CNN/output2.csv', index=False, header=False)

print(f"处理完成，结果已保存到 output.csv")
