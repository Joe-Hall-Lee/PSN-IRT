import pandas as pd
import numpy as np
import os

# --- 配置 ---
FULL_DATA_PATH = "data/combine.csv"  # 完整 0/1 响应数据矩阵
OUTPUT_HALF_DATA_PATH = "data/percent_0.7.csv"
OUTPUT_INDICES_PATH = "results/selected_item_indices_0.7.csv"  # 保存被选中题目的原始索引
SPLIT_RATIO = 0.7
RANDOM_SEED = 42  # 确保每次随机抽样的结果都一样


def create_random_item_subset(full_data_path, output_data_path, output_indices_path, ratio, seed):
    """
    从完整的响应矩阵中随机抽取一部分题目（列），
    保存新的数据文件，并同时保存被选中列的原始索引。
    """
    print(f"正在从 {full_data_path} 加载完整数据...")
    try:
        df = pd.read_csv(full_data_path, header=None)
    except FileNotFoundError:
        print(f"错误: 文件 '{full_data_path}' 未找到。")
        return

    num_total_items = df.shape[1]
    num_items_to_select = int(num_total_items * ratio)

    print(
        f"总共有 {num_total_items} 道题目。将随机抽取 {num_items_to_select} 道 ({ratio*100:.0f}%)。")

    # 设置随机种子以保证结果可复现
    np.random.seed(seed)

    # 从所有列索引中不重复地随机选择
    all_column_indices = np.arange(num_total_items)
    selected_column_indices = np.random.choice(
        all_column_indices, size=num_items_to_select, replace=False)

    # 保存被选中的索引
    indices_df = pd.DataFrame({'selected_index': selected_column_indices})
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_indices_path), exist_ok=True)
    indices_df.to_csv(output_indices_path, index=False)
    print(
        f"已将被选中的 {len(selected_column_indices)} 个题目的原始索引保存至: {output_indices_path}")

    # 提取对应的列并保存
    df_subset = df.iloc[:, selected_column_indices]
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df_subset.to_csv(output_data_path, header=False, index=False)
    print(f"数据文件保存至: {output_data_path}")


if __name__ == "__main__":
    create_random_item_subset(
        FULL_DATA_PATH, OUTPUT_HALF_DATA_PATH, OUTPUT_INDICES_PATH, SPLIT_RATIO, RANDOM_SEED)
