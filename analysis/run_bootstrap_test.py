# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import wilcoxon
from tqdm import tqdm

MODEL_A_PREDICTIONS_PATH = "results/test_predictions_psn_irt.csv"

MODEL_B_PREDICTIONS_PATH = "results/test_predictions_deep_irt.csv"
# 检验的性能指标。可以选择 f1_score 或 accuracy_score
METRIC_TO_TEST = f1_score
METRIC_NAME = "F1 Score"
# Bootstrap 重采样的次数
N_BOOTSTRAPS = 1000


def run_bootstrap_test(df_a, df_b):
    """
    执行自助法检验 (Bootstrap Test) 来判断模型 A 是否显著优于模型 B。
    """
    # 确保两个文件的真实标签是一致的
    if not np.array_equal(df_a['ground_truth'], df_b['ground_truth']):
        raise ValueError("错误：两个预测文件中的 'ground_truth' 列不匹配。")

    labels = df_a['ground_truth'].values
    preds_a = df_a['prediction'].values
    preds_b = df_b['prediction'].values

    n_samples = len(labels)

    # 计算在原始完整测试集上的性能差异
    original_score_a = METRIC_TO_TEST(labels, preds_a)
    original_score_b = METRIC_TO_TEST(labels, preds_b)
    original_diff = original_score_a - original_score_b

    print(f"原始性能 ({METRIC_NAME}):")
    print(f"  - PSN-IRT (模型A): {original_score_a:.4f}")
    print(f"  - Deep-IRT (模型B): {original_score_b:.4f}")
    print(f"  - 原始性能差异 (A - B): {original_diff:.4f}")

    # 2. 执行 Bootstrap 重采样
    count_b_is_better_or_equal = 0
    for _ in tqdm(range(N_BOOTSTRAPS), desc="执行 Bootstrap 检验"):
        # 通过有放回的随机抽样，创建新的索引列表
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # 根据新索引创建重采样后的数据集
        resampled_labels = labels[indices]
        resampled_preds_a = preds_a[indices]
        resampled_preds_b = preds_b[indices]

        # 在重采样的数据上计算性能
        score_a = METRIC_TO_TEST(resampled_labels, resampled_preds_a)
        score_b = METRIC_TO_TEST(resampled_labels, resampled_preds_b)

        # 统计基线模型表现更好或持平的次数
        if score_b >= score_a:
            count_b_is_better_or_equal += 1

    # 3. 计算 p-value
    # p-value 是在“两个模型性能无真实差异”的原假设下，观察到基线模型 B 表现优于或等于 A 的概率
    p_value = (count_b_is_better_or_equal + 1) / (N_BOOTSTRAPS + 1)

    print("\n--- 自助法检验 (Bootstrap Test) 结果 ---")
    print(f"检验指标: {METRIC_NAME}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("结论: PSN-IRT 的性能提升在统计上是显著的 (p < 0.05)。")
    else:
        print("结论: PSN-IRT 的性能提升在统计上不显著 (p >= 0.05)。")


def run_wilcoxon_test(df_a, df_b):
    """
    执行威尔科克森符号秩检验 (Wilcoxon Signed-rank Test)。
    这个检验比较的是两个配对样本的差异是否来自一个中位数为 0 的分布。
    """
    labels = df_a['ground_truth'].values
    preds_a = df_a['prediction'].values
    preds_b = df_b['prediction'].values

    # 为每个样本点计算一个“得分”，例如，1代表预测正确，0代表错误
    scores_a = (preds_a == labels).astype(int)
    scores_b = (preds_b == labels).astype(int)

    # 计算两个模型得分的差异
    differences = scores_a - scores_b

    # 移除差异为 0 的样本点，因为它们对检验没有贡献
    differences = differences[differences != 0]

    if len(differences) < 10:  # 如果非零差异样本太少，检验可能不可靠
        print("\n--- 威尔科克森符号秩检验 ---")
        print("警告：非零差异的样本点过少，检验结果可能不可靠。")
        return

    # 执行检验
    stat, p_value = wilcoxon(differences, alternative='greater')

    print("\n--- 威尔科克森符号秩检验 (Wilcoxon Signed-rank Test) 结果 ---")
    print(f"检验指标: 逐项准确率差异")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("结论: PSN-IRT 的逐项表现显著优于 Deep-IRT (p < 0.05)。")
    else:
        print("结论: PSN-IRT 的逐项表现没有显著优于 Deep-IRT (p >= 0.05)。")


if __name__ == "__main__":
    try:
        df_a = pd.read_csv(MODEL_A_PREDICTIONS_PATH)
        df_b = pd.read_csv(MODEL_B_PREDICTIONS_PATH)

        # 执行两种检验
        run_bootstrap_test(df_a, df_b)
        run_wilcoxon_test(df_a, df_b)

    except FileNotFoundError as e:
        print(
            f"错误: 找不到预测文件。请确保 '{MODEL_A_PREDICTIONS_PATH}' 和 '{MODEL_B_PREDICTIONS_PATH}' 文件都存在。")
        print(f"详细信息: {e}")
