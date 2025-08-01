import pandas as pd
from scipy.stats import kendalltau

file1_path = 'results/student_abilities_cluster_semantic_4pl_part1.csv'
file2_path = 'results/student_abilities_cluster_semantic_4pl_part2.csv'

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# 检查是否包含 'ability' 列
if 'ability' not in df1.columns or 'ability' not in df2.columns:
    raise ValueError("Both files must contain an 'ability' column.")

if 'student_id' in df1.columns and 'student_id' in df2.columns:
    df1 = df1.set_index('student_id')
    df2 = df2.set_index('student_id')
    # 对齐两个 DataFrame
    df1, df2 = df1.align(df2, join='inner', axis=0)

# 提取 ability 列
ability1 = df1['ability']
ability2 = df2['ability']

# 检查两列长度是否一致
if len(ability1) != len(ability2):
    raise ValueError(
        "The 'ability' columns in the two files are not of the same length after alignment.")

# 计算肯德尔相关系数
tau, p_value = kendalltau(ability1, ability2)

# 输出结果
print(f"Kendall's Tau: {tau}")
print(f"P-value: {p_value}")

# 打印按照 ability 从大到小排序的 student_id 顺序
sorted_df1 = df1.sort_values(by='ability', ascending=False)  # 按 ability 降序排序
sorted_df2 = df2.sort_values(by='ability', ascending=False)  # 按 ability 降序排序

print("\nSorted student_id order in part1 (by ability descending):")
print(sorted_df1.index.tolist())  # 打印 part1 的 student_id 顺序

print("\nSorted student_id order in part2 (by ability descending):")
print(sorted_df2.index.tolist())  # 打印 part2 的 student_id 顺序
