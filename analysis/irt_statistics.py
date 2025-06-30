# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def calculate_icc(ability, difficulty, discrimination, guessing, feasibility):
    """计算四参数 Logistic 模型的项目特征曲线 (ICC)。"""
    exponent = discrimination * (ability - difficulty)
    # 增加对 np.exp(-exponent) 结果的检查，防止溢出为 inf
    exp_val = np.exp(-exponent)
    if np.isinf(exp_val):  # 如果指数结果过大，分母为 inf，icc 趋向于 guessing
        return guessing
    icc = guessing + (feasibility - guessing) / (1 + exp_val)
    return icc


def calculate_leh_at_ability(difficulty, discrimination, guessing, feasibility, ability=3.7716236):
    """计算在给定能力值处的局部估计 Headroom (LEH) 分数。"""
    icc_at_ability = calculate_icc(
        ability, difficulty, discrimination, guessing, feasibility)
    if discrimination == 0 or (feasibility - guessing) == 0:
        return 0, icc_at_ability
    delta = 0.0001
    icc_slightly_lower = calculate_icc(
        ability - delta, difficulty, discrimination, guessing, feasibility)
    leh = (icc_at_ability - icc_slightly_lower) / delta
    if np.abs(leh) > 1e5:
        pass
    return leh, icc_at_ability


def calculate_fisher_information_at_ability(discrimination, p_theta, guessing, feasibility):
    """计算在给定能力值 theta 处的 Fisher 信息。"""
    if (feasibility - guessing) == 0 or p_theta <= guessing or p_theta >= feasibility:
        return 0
    if p_theta == 0 or p_theta == 1:
        if (guessing == 0 and p_theta == 0) or (feasibility == 1 and p_theta == 1):
            pass
        else:
            return 0
    numerator = discrimination**2 * \
        (p_theta - guessing)**2 * (feasibility - p_theta)**2
    denominator_factor = p_theta * (1 - p_theta)
    if denominator_factor == 0:
        return 0
    denominator = (feasibility - guessing)**2 * denominator_factor
    if denominator == 0:
        return 0
    return numerator / denominator


def analyze_irt_results(item_file="item_parameters.csv",
                        output_file="dataset_analysis.csv",
                        plot_boxplots=False,
                        ability_for_leh_fisher=3.7716236):
    try:
        item_df = pd.read_csv(item_file)
    except FileNotFoundError as e:
        print(f"错误: 输入文件未找到 - {e.filename}")
        return None  # 返回 None 表示函数执行失败
    except pd.errors.EmptyDataError as e:
        print(f"错误: 输入文件为空 - {e.filename}")
        return None
    except Exception as e:
        print(f"读取输入文件时发生错误: {e}")
        return None


    required_columns = ['difficulty',
                        'discrimination', 'guessing', 'feasibility']
    if not all(col in item_df.columns for col in required_columns):
        print(
            f"错误: Item file '{item_file}' 必须包含列: {required_columns}. 当前列: {item_df.columns.tolist()}")
        return None

    sorted_indices = sorted(DATASET_INDICES.items(), key=lambda x: x[1])
    dataset_ranges = []
    actual_end_df = len(item_df) - 1
    for i in range(len(sorted_indices)):
        dataset_name, start = sorted_indices[i]
        next_start = sorted_indices[i +
                                    1][1] if i < len(sorted_indices)-1 else len(item_df)
        end = min(next_start - 1, actual_end_df)
        if start <= actual_end_df and start <= end:
            dataset_ranges.append((dataset_name, start, end))
        else:
            print(
                f"警告: 数据集 '{dataset_name}' 的索引范围 [{start}-{end}] 无效或部分超出数据实际范围 {actual_end_df}。跳过此数据集。")

    if not dataset_ranges:
        print("错误：没有有效的数据集范围可供分析。")
        return None

    analysis_results = []
    all_datasets_for_plot = pd.DataFrame()  # 为箱线图准备数据

    for dataset_name, start_idx, end_idx in dataset_ranges:
        dataset_items = item_df.iloc[start_idx:end_idx + 1].copy()
        if dataset_items.empty:
            print(f"警告: 数据集 '{dataset_name}' 在切片后为空。跳过。")
            continue

        leh_p_theta = dataset_items.apply(lambda row: calculate_leh_at_ability(
            row['difficulty'], row['discrimination'], row['guessing'], row['feasibility'], ability=ability_for_leh_fisher
        ), axis=1, result_type='expand')
        dataset_items['LEH'] = leh_p_theta[0]
        dataset_items['P_at_ability'] = leh_p_theta[1]

        dataset_items['Fisher Information'] = dataset_items.apply(lambda row: calculate_fisher_information_at_ability(
            row['discrimination'], row['P_at_ability'], row['guessing'], row['feasibility']
        ), axis=1)

        # 为箱线图数据添加数据集名称列
        dataset_items_for_plot_slice = dataset_items.copy()
        dataset_items_for_plot_slice['Dataset'] = dataset_name
        all_datasets_for_plot = pd.concat(
            [all_datasets_for_plot, dataset_items_for_plot_slice])

        item_stats = {
            "Dataset": dataset_name,
            "Difficulty": dataset_items['difficulty'].mean(),
            "Discrimination": dataset_items['discrimination'].mean(),
            "Guessing Rate": dataset_items['guessing'].mean(),
            "Feasibility": dataset_items['feasibility'].mean(),
            "LEH": dataset_items['LEH'].mean(),
            "Fisher Information": dataset_items['Fisher Information'].mean()
        }
        analysis_results.append(item_stats)

    result_df = pd.DataFrame(analysis_results)
    if result_df.empty:
        print("错误：未能生成任何分析结果。")
        return None

    result_df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")

    rank_df = result_df.copy()
    if not rank_df.empty:
        rank_df['Difficulty_Rank'] = rank_df['Difficulty'].rank(
            ascending=False)
        rank_df['Discrimination_Rank'] = rank_df['Discrimination'].rank(
            ascending=False)

        if 'Guessing Rate' in rank_df.columns:  # 检查列是否存在
            rank_df['Guessing_Rank'] = rank_df['Guessing Rate'].rank(ascending=True)
        else:
            print("警告：Guessing Rate 列不存在，无法计算 Guessing_Rank。")

        rank_df['Feasibility_Rank'] = rank_df['Feasibility'].rank(
            ascending=False)
        rank_df['LEH_Rank'] = rank_df['LEH'].rank(ascending=False)
        rank_df['Fisher Information_Rank'] = rank_df['Fisher Information'].rank(
            ascending=False)

        # 确保所有参与 Total_Rank 计算的列都存在
        rank_cols_for_total = ['Discrimination_Rank', 'Guessing_Rank',
                               'Feasibility_Rank', 'LEH_Rank', 'Fisher Information_Rank']
        existing_rank_cols = [
            col for col in rank_cols_for_total if col in rank_df.columns]

        if len(existing_rank_cols) == len(rank_cols_for_total):
            rank_df['Total_Rank'] = rank_df[existing_rank_cols].sum(axis=1)
            rank_df = rank_df.sort_values(by='Total_Rank')
        else:
            print(
                f"警告: 由于部分排名列缺失，无法计算 Total_Rank。缺失的列可能为: {set(rank_cols_for_total) - set(existing_rank_cols)}")

        rank_output_file = output_file.replace(".csv", "_rankings.csv")
        rank_df.to_csv(rank_output_file, index=False)
        print(f"Ranking results saved to {rank_output_file}")
    else:
        print("警告：result_df 为空，跳过排名计算。")

    if plot_boxplots:
        if all_datasets_for_plot.empty or 'Dataset' not in all_datasets_for_plot.columns:
            print("错误: 用于箱线图的数据 (all_datasets_for_plot) 为空或缺少 'Dataset' 列。无法绘制。")
        else:
            sns.set(style="whitegrid")

            num_boxplot_rows = 2
            num_boxplot_cols = 3

            fig_width = num_boxplot_cols * 5  # 每个子图宽度增加，以容纳更大的字体
            fig_height = num_boxplot_rows * 4  # 每个子图高度增加
            fig, axes = plt.subplots(
                num_boxplot_rows, num_boxplot_cols, figsize=(fig_width, fig_height))

            params_to_plot = ['difficulty', 'discrimination',
                              'guessing', 'feasibility', 'LEH', 'Fisher Information']
            # 确保 dataset_order 只包含实际存在于 all_datasets_for_plot 中的数据集名称
            unique_datasets_in_plot_data = all_datasets_for_plot['Dataset'].unique(
            )
            dataset_order = [
                name for name, _, _ in dataset_ranges if name in unique_datasets_in_plot_data]

            ax_flat = axes.flatten()

            for i, param in enumerate(params_to_plot):
                if i >= len(ax_flat):
                    break  # 安全检查
                ax = ax_flat[i]

                sns.boxplot(ax=ax, x='Dataset', y=param,
                            data=all_datasets_for_plot, palette="Set3", order=dataset_order)

                title = param.capitalize().replace('_', ' ')
                if param == 'LEH':
                    title = 'LEH'
                elif param == 'Fisher Information':
                    title = 'Fisher Information'
                elif param == 'guessing':
                    title = 'Guessing Rate'
                elif param == 'discrimination':
                    title = 'Discriminability'

                ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"])
                # X 轴刻度标签（数据集名称）
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                                   fontsize=plt.rcParams["xtick.labelsize"] - 2)

                # Y 轴刻度标签（参数值）
                # 与 X 轴刻度标签字号协调
                ax.tick_params(
                    axis='y', labelsize=plt.rcParams["xtick.labelsize"] - 2)

                # Y轴标题（参数名称）
                ax.set_ylabel(title, fontsize=plt.rcParams["axes.labelsize"])

                current_row = i // num_boxplot_cols
                if current_row == (num_boxplot_rows - 1):  # 如果是最后一行
                    ax.set_xlabel(
                        'Dataset', fontsize=plt.rcParams["axes.labelsize"])
                else:
                    ax.set_xlabel('')

            # 隐藏任何多余的子图
            for j in range(len(params_to_plot), num_boxplot_rows * num_boxplot_cols):
                if j < len(ax_flat):
                    fig.delaxes(ax_flat[j])

            plt.tight_layout(pad=0.8)
            try:
                plt.savefig("images/benchmark_analysis.png", dpi=300)
                print("箱线图已保存。")
            except Exception as e:
                print(f"保存箱线图时发生错误: {e}")
            plt.show()

    return result_df if 'result_df' in locals() and result_df is not None else None


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 15

    # 调用时传递 plot_boxplots=True
    df_results = analyze_irt_results(
        plot_boxplots=True, item_file="results/item_parameters.csv")
    if df_results is not None and not df_results.empty:
        print("\nSummary Analysis DataFrame Head:")
        print(df_results.head())
    else:
        print("未能成功执行 analyze_irt_results 或结果为空。")
