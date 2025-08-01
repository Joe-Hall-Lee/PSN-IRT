import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --- 配置 ---
PARAMS_FULL_PATH = "results/item_parameters.csv"
PARAMS_HALF_PATH = "results/item_parameters_0.3.csv"

INDICES_PATH = "results/selected_item_indices_0.3.csv"

# 需要进行相关性分析的参数列名
PARAMETER_COLUMNS = ['difficulty', 'discrimination', 'guessing', 'feasibility']


def analyze_parameter_stability_random(params_full_path, params_half_path, indices_path, param_columns):
    """
    使用索引文件对齐随机抽样的参数，并计算相关性。
    """
    print("加载参数文件和索引文件...")
    try:
        df_full = pd.read_csv(params_full_path)
        df_half = pd.read_csv(params_half_path)
        df_indices = pd.read_csv(indices_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件。请确保所有文件都已生成并位于正确路径。")
        print(f"详细信息: {e}")
        return

    # 获取被选中的题目的原始索引列表
    selected_indices = df_indices['selected_index'].tolist()

    print(f"根据索引文件，共有 {len(selected_indices)} 个项目被随机抽中进行训练。")
    if len(df_half) != len(selected_indices):
        print(
            f"警告: 'half'参数文件 ({len(df_half)}行) 与索引文件 ({len(selected_indices)}行) 的长度不匹配！")

    # 从“完整”参数文件中，只抽取出那些被选中的行
    df_full_subset = df_full.iloc[selected_indices].copy()

    # df_full_subset 和 df_half 的行应该是一一对应的
    # 为了安全拼接，重置它们的索引
    df_full_subset.reset_index(drop=True, inplace=True)
    df_half.reset_index(drop=True, inplace=True)

    print("\n--- 参数稳定性相关性分析结果 (随机抽样) ---")

    for param in param_columns:
        if param in df_full_subset.columns and param in df_half.columns:
            # 将两组参数放在一起以便分析
            analysis_data = pd.DataFrame({
                'full_run': df_full_subset[param],
                'half_run': df_half[param]
            }).dropna()

            if len(analysis_data) < 2:
                print(f"\n对于参数 '{param}'，有效数据点不足。")
                continue

            # 计算相关系数
            p_corr, p_p_value = pearsonr(
                analysis_data['full_run'], analysis_data['half_run'])
            s_corr, s_p_value = spearmanr(
                analysis_data['full_run'], analysis_data['half_run'])

            print(f"\n参数 '{param}' 的稳定性:")
            print(f"  - 皮尔逊相关系数: {p_corr:.4f} (p-value: {p_p_value:.4f})")
            print(f"  - 斯皮尔曼相关系数: {s_corr:.4f} (p-value: {s_p_value:.4f})")
        else:
            print(f"\n警告: 未能在两个参数文件中都找到参数 '{param}' 的对应列。")


if __name__ == "__main__":
    analyze_parameter_stability_random(
        PARAMS_FULL_PATH, PARAMS_HALF_PATH, INDICES_PATH, PARAMETER_COLUMNS)
