# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 数据集索引配置
DATASET_INDICES = {
    "ARC-C": 0,
    "BBH": 295,
    "Chinese SimpleQA": 6806,
    "GPQA Diamond": 9806,
    "GSM8K": 10004,
    "HellaSwag": 11323,
    "HumanEval": 21365,
    "MATH": 21529,
    "MBPP": 26529,
    "MMLU": 27029,
    "TheoremQA": 41071
}

def plot_difficulty_discrimination(item_file="results/item_parameters.csv"):
    # 读取数据
    try:
        item_df = pd.read_csv(item_file)
    except FileNotFoundError:
        print(f"错误: 文件 '{item_file}' 未找到。")
        return
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{item_file}' 为空。")
        return
    except Exception as e:
        print(f"读取文件 '{item_file}' 时发生错误: {e}")
        return

    # 验证必需的列是否存在
    required_cols = ['discrimination', 'difficulty'] 
    if not all(col in item_df.columns for col in required_cols):
        print(f"错误: 文件 '{item_file}' 必须包含列: {required_cols}。实际列名: {item_df.columns.tolist()}")
        return

    # 确定数据集边界
    sorted_indices = sorted(DATASET_INDICES.items(), key=lambda x: x[1])
    dataset_ranges = []
    actual_end_df = len(item_df) - 1
    for i in range(len(sorted_indices)):
        dataset_name, start = sorted_indices[i]
        next_start = sorted_indices[i+1][1] if i < len(sorted_indices)-1 else len(item_df)
        end = min(next_start - 1, actual_end_df)
        
        if start > actual_end_df:
            print(f"警告: 数据集 '{dataset_name}' 的起始索引 {start} 超出项目数据范围 {actual_end_df}。跳过此数据集。")
            continue
        if start > end:
             print(f"警告: 数据集 '{dataset_name}' 的范围无效 (start={start}, end={end})。可能数据不足。跳过此数据集。")
             continue
        dataset_ranges.append((dataset_name, start, end))

    if not dataset_ranges:
        print("没有有效的数据集范围可以处理。")
        return
        
    num_datasets = len(dataset_ranges)
    num_rows = 2
    num_cols = (num_datasets + num_rows - 1) // num_rows # 计算每行需要的列数
    
    subplot_factor = 3
    fig, axes = plt.subplots(num_rows, num_cols, 
                             figsize=(subplot_factor * num_cols, subplot_factor * num_rows + 0.5), # 略微增加总高度以容纳 X 轴标签
                             sharex=False, sharey=True)
    
    sns.set(style="whitegrid")

    title_fontsize = 14
    axis_label_fontsize = 14
    tick_label_fontsize = 14
    scatter_point_size = 5 


    # 遍历每个数据集绘制点状图
    for idx, (dataset_name, start_idx, end_idx) in enumerate(dataset_ranges):
        dataset_items = item_df.iloc[start_idx:end_idx+1]
        
        if dataset_items.empty:
            print(f"警告: 数据集 '{dataset_name}' 在索引范围 [{start_idx}, {end_idx}] 内没有数据。将处理下一个。")
            # 隐藏这个空子图
            row_idx_empty = idx // num_cols
            col_idx_empty = idx % num_cols
            if num_rows == 1 and num_cols == 1: fig_ax_to_del = axes
            elif num_rows == 1 : fig_ax_to_del = axes[col_idx_empty]
            elif num_cols == 1: fig_ax_to_del = axes[row_idx_empty]
            else: fig_ax_to_del = axes[row_idx_empty, col_idx_empty]
            if fig_ax_to_del is not None: fig.delaxes(fig_ax_to_del)
            continue

        row_idx = idx // num_cols
        col_idx = idx % num_cols

        if num_rows == 1 and num_cols == 1: ax = axes
        elif num_rows == 1 : ax = axes[col_idx]
        elif num_cols == 1: ax = axes[row_idx]
        else: ax = axes[row_idx, col_idx]
        
        sns.scatterplot(data=dataset_items, x='discrimination', y='difficulty', ax=ax,
                        color=sns.color_palette("husl", num_datasets)[idx % len(sns.color_palette("husl", num_datasets))], 
                        s=scatter_point_size) 
        
        ax.set_title(f'{dataset_name}', fontsize=title_fontsize, pad=3)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        
        if col_idx == 0: # 只在第一列显示 Y 轴标签
            ax.set_ylabel('Difficulty', fontsize=axis_label_fontsize)
        else:
            ax.set_ylabel('') # 其他列不显示 Y 轴标签
        if row_idx == num_rows - 1:
            ax.set_xlabel('Discriminability', fontsize=axis_label_fontsize)
            ax.tick_params(axis='x', bottom=True, labelbottom=True)  # 确保显示刻度
        elif col_idx == num_cols - 1:
            ax.set_xlabel('Discriminability', fontsize=axis_label_fontsize)
            ax.tick_params(axis='x', bottom=True, labelbottom=True)  # 确保显示刻度
        else:  # 其他情况
            ax.set_xlabel('')
            ax.tick_params(axis='x', bottom=True, labelbottom=False)  # 隐藏刻度
        
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1, 4) 
        ax.set_ylim(-3, 2)

    # 隐藏多余的子图
    for i in range(num_datasets, num_rows * num_cols):
        row_idx_empty = i // num_cols
        col_idx_empty = i % num_cols
        if num_rows == 1 and num_cols == 1: continue # 只有一个子图且已绘制
        elif num_rows == 1 : fig_ax_to_del = axes[col_idx_empty]
        elif num_cols == 1: fig_ax_to_del = axes[row_idx_empty]
        else: fig_ax_to_del = axes[row_idx_empty, col_idx_empty]
        if fig_ax_to_del is not None: fig.delaxes(fig_ax_to_del)

    # 调整布局
    plt.tight_layout(w_pad=0.5, h_pad=0.8)


    # 保存图形
    try:
        plt.savefig('images/diff_disc.png', dpi=300)
    except Exception as e:
        print(f"保存图形时发生错误: {e}")

    plt.show()

if __name__ == "__main__":
    plot_difficulty_discrimination()