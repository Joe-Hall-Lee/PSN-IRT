# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from scipy.stats import kendalltau
from sklearn.cluster import KMeans

# 数据集索引配置
DATASET_INDICES = {
    "ARC-C": 0,
    "BBH": 295,
    "Chinese SimpleQA": 6806,
    "GPQA Diamond": 9806,
    "GSM8K": 10004,
    "HellaSwag": 11323,
    "HumanEval": 21529,
    "MATH": 26529,
    "MBPP": 27029,
    "MMLU": 41071,
    "TheoremQA": 41871
}

# 模型列表
model_list = ['360GPT2-Pro', 'DeepSeek-V3', 'Doubao-Pro', 'Gemini-1.5', 'Gemma-2B',
              'Hunyuan-Turbo', 'Mistral-7B', 'Moonshot-v1', 'Qwen-Plus', 'Qwen2.5-3B',
              'Vicuna-7B', 'Yi-Lightning']
open_source_models = ['Mistral-7B', 'Qwen2.5-3B', 'Vicuna-7B', 'Gemma-2B']
leading_models = [
    model for model in model_list if model not in open_source_models]

# 手动参考排名
manual_reference_ranking = {
    'DeepSeek-V3': 1,
    'Hunyuan-Turbo': 2,
    'Yi-Lightning': 3,
    'Qwen-Plus': 4,
    'Gemini-1.5': 5,
    '360GPT2-Pro': 6,
    'Doubao-Pro': None,
    'Moonshot-v1': 7,
    'Qwen2.5-3B': None,
    'Mistral-7B': 10,
    'Vicuna-7B': 11,
    'Gemma-2B': 12
}


def evaluate_model_accuracy_on_subset(model_responses_df, item_indices, model_list):
    """评估模型在指定题目子集上的准确率。"""
    model_accuracies = {}
    for i, model in enumerate(model_list):
        if 0 <= i < model_responses_df.shape[0]:
            valid_item_indices = [
                idx for idx in item_indices if idx < model_responses_df.shape[1]]
            if not valid_item_indices:
                print(f"警告：对于模型 '{model}'，在所选 item_indices 中没有有效的列索引。")
                model_accuracies[model] = np.nan
                continue
            model_responses_subset = model_responses_df.iloc[i,
                                                             valid_item_indices]
            if model_responses_subset.empty:
                model_accuracies[model] = np.nan
            else:
                accuracy = model_responses_subset.mean().round(4)
                model_accuracies[model] = accuracy
        else:
            model_accuracies[model] = np.nan
            print(f"警告：模型列表中的模型 '{model}' 在 combine.csv 中没有对应的行 ({i})。")
    return model_accuracies


def select_items_by_clustering(model_responses_df, num_items, random_seed=42):
    """基于题目在模型上的 0/1 结果向量使用 K-Means 聚类选择 num_items 个题目（每个簇选一个）。"""
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 获取题目特征：每个题目的 0/1 结果向量（模型数维）
    response_vectors = model_responses_df.to_numpy().T  # 转置，列为题目，行为模型
    print(f"response_vectors shape: {response_vectors.shape}")

    # 检查 NaN
    if np.any(np.isnan(response_vectors)):
        nan_indices = np.where(np.isnan(response_vectors))
        print(
            f"错误：response_vectors 包含 NaN，位置：{list(zip(nan_indices[0], nan_indices[1]))}")
        print(f"NaN 样本示例：{response_vectors[nan_indices[0][:5]]}")
        raise ValueError("response_vectors 包含 NaN，请检查 combine.csv 数据。")

    # 检查唯一值，确保是 0/1
    unique_values = np.unique(response_vectors)
    print(f"response_vectors 唯一值：{unique_values}")
    if not set(unique_values).issubset({0, 1}):
        print(f"警告：response_vectors 包含非 0/1 值：{unique_values}")
        # 强制转换为 0/1（假设非 0/1 值是异常，转换为 0）
        response_vectors = np.where(
            np.isin(response_vectors, [0, 1]), response_vectors, 0)
        print("已将非 0/1 值转换为 0。")

    # 检查题目数量
    if response_vectors.shape[0] < num_items:
        print(f"错误：题目数量 ({response_vectors.shape[0]}) 小于目标聚类数量 ({num_items})。")
        return []

    # K-Means 聚类，簇数等于目标题目数量
    kmeans = KMeans(n_clusters=num_items, random_state=random_seed)
    cluster_labels = kmeans.fit_predict(response_vectors)
    cluster_centers = kmeans.cluster_centers_

    # 从每个簇中选择距离簇中心最近的题目
    selected_items = []
    for cluster in range(num_items):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) == 0:
            print(f"警告：簇 {cluster} 为空，跳过。")
            continue
        # 计算每个题目到簇中心的距离
        distances = np.linalg.norm(
            response_vectors[cluster_indices] - cluster_centers[cluster], axis=1)
        # 选择距离最近的题目
        selected_item = cluster_indices[np.argmin(distances)]
        selected_items.append(selected_item)

    # 检查选择数量
    if len(selected_items) < num_items:
        print(f"警告：仅选择了 {len(selected_items)} 个题目，小于目标 {num_items}，可能由于空簇。")

    return sorted(selected_items)


def evaluate_clustering_selection(model_responses_df, num_items, model_list, leading_models, reference_ranking, random_seed=42):
    """评估基于聚类的题目选择，计算方差和 Kendall's $\tau$。"""
    # 选择题目
    selected_item_indices = select_items_by_clustering(
        model_responses_df, num_items, random_seed)

    # 计算所有模型的准确率（w/ weak models）
    model_accuracies = evaluate_model_accuracy_on_subset(
        model_responses_df, selected_item_indices, model_list)

    # 筛选领先模型（w/o weak models）
    leading_accuracies = {model: acc for model, acc in model_accuracies.items(
    ) if model in leading_models and not np.isnan(acc)}

    # 计算方差
    all_accuracies = [acc for acc in model_accuracies.values()
                      if not np.isnan(acc)]
    leading_accuracies_values = [
        acc for acc in leading_accuracies.values() if not np.isnan(acc)]
    variance_all = np.var(all_accuracies) if len(
        all_accuracies) > 1 else np.nan
    variance_leading = np.var(leading_accuracies_values) if len(
        leading_accuracies_values) > 1 else np.nan

    # 计算 Kendall's $\tau$
    # w/ weak models
    sorted_accuracies_all = sorted([(model, acc) for model, acc in model_accuracies.items(
    ) if not np.isnan(acc)], key=lambda x: x[1], reverse=True)
    accuracy_ranking_all = {model: rank for rank,
                            (model, _) in enumerate(sorted_accuracies_all, 1)}
    common_models_all = [
        m for m in model_list if m in accuracy_ranking_all and m in reference_ranking and reference_ranking[m] is not None]
    if len(common_models_all) < 2:
        print(
            f"警告：w/ weak models 没有足够的共同模型（{len(common_models_all)}）来计算 Kendall's $\tau$。")
        tau_all, p_all = np.nan, np.nan
    else:
        accuracy_ranks_all = pd.Series(
            accuracy_ranking_all).reindex(common_models_all)
        ref_ranks_all = pd.Series(reference_ranking).reindex(common_models_all)
        tau_all, p_all = kendalltau(accuracy_ranks_all, ref_ranks_all)

    # w/o weak models
    sorted_accuracies_leading = sorted(
        [(model, acc) for model, acc in leading_accuracies.items()], key=lambda x: x[1], reverse=True)
    accuracy_ranking_leading = {
        model: rank for rank, (model, _) in enumerate(sorted_accuracies_leading, 1)}
    common_models_leading = [
        m for m in leading_models if m in accuracy_ranking_leading and m in reference_ranking and reference_ranking[m] is not None]
    if len(common_models_leading) < 2:
        print(
            f"警告：w/o weak models 没有足够的共同模型（{len(common_models_leading)}）。")
        tau_leading, p_leading = np.nan, np.nan
    else:
        accuracy_ranks_leading = pd.Series(
            accuracy_ranking_leading).reindex(common_models_leading)
        ref_ranks_leading = pd.Series(
            reference_ranking).reindex(common_models_leading)
        tau_leading, p_leading = kendalltau(
            accuracy_ranks_leading, ref_ranks_leading)

    return {
        'num_items': num_items,
        'variance_all': variance_all,
        'tau_all': tau_all,
        'p_all': p_all,
        'num_common_all': len(common_models_all),
        'variance_leading': variance_leading,
        'tau_leading': tau_leading,
        'p_leading': p_leading,
        'num_common_leading': len(common_models_leading)
    }


if __name__ == "__main__":
    # 设置随机种子
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 加载数据
    try:
        model_responses_df = pd.read_csv("data/combine.csv", header=None)
        num_total_items = model_responses_df.shape[1]
        print(f"combine.csv 形状：{model_responses_df.shape}")
        # 检查 NaN
        if model_responses_df.isna().any().any():
            nan_locations = model_responses_df.isna(
            ).stack()[model_responses_df.isna().stack()]
            print(f"错误：combine.csv 包含 NaN，位置：{nan_locations.index.tolist()}")
            raise ValueError("combine.csv 包含 NaN，请检查数据完整性。")
        # 检查唯一值
        unique_values = np.unique(model_responses_df.to_numpy())
        print(f"combine.csv 唯一值：{unique_values}")
        if not set(unique_values).issubset({0, 1}):
            print(f"警告：combine.csv 包含非 0/1 值：{unique_values}")
            # 强制转换为 0/1
            model_responses_df = model_responses_df.where(
                model_responses_df.isin([0, 1]), 0)
            print("已将非 0/1 值转换为 0。")
    except FileNotFoundError as e:
        print(f"错误：找不到模型作答文件 {e.filename}，无法进行评估。")
        exit()
    except Exception as e:
        print(f"读取 combine.csv 时发生错误: {e}")
        exit()

    # 计算所有模型的整体准确率（w/ weak models）
    accuracies_all = {}
    if model_responses_df.shape[0] >= len(model_list):
        if model_responses_df.shape[0] > len(model_list):
            print(
                f"警告：combine.csv 的行数 ({model_responses_df.shape[0]}) 大于模型列表长度 ({len(model_list)})。仅使用前 {len(model_list)} 行。")
        for i, model in enumerate(model_list):
            accuracies_all[model] = model_responses_df.iloc[i,
                                                            :].mean().round(4)
    else:
        print(
            f"错误：combine.csv 的行数 ({model_responses_df.shape[0]}) 小于模型列表长度 ({len(model_list)})。")
        for i in range(model_responses_df.shape[0]):
            if i < len(model_list):
                accuracies_all[model_list[i]
                               ] = model_responses_df.iloc[i, :].mean().round(4)

    print("\n所有模型在所有题目上的准确率:")
    sorted_accuracies_all = sorted(
        accuracies_all.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, accuracy) in enumerate(sorted_accuracies_all, 1):
        print(f"  排名 {rank}: {model} - 准确率: {accuracy}")

    accuracy_ranking_all = {model: rank for rank,
                            (model, _) in enumerate(sorted_accuracies_all, 1)}
    common_models_all = [
        m for m in model_list if m in accuracy_ranking_all and m in manual_reference_ranking and manual_reference_ranking[m] is not None]
    if len(common_models_all) < 2:
        print(
            f"警告：w/ weak models 没有足够的共同模型（{len(common_models_all)}）来计算 Kendall's $\tau$。")
        tau_all, p_all = np.nan, np.nan
    else:
        accuracy_ranks_all = pd.Series(
            accuracy_ranking_all).reindex(common_models_all)
        ref_ranks_all = pd.Series(
            manual_reference_ranking).reindex(common_models_all)
        tau_all, p_all = kendalltau(accuracy_ranks_all, ref_ranks_all)

    accuracies_for_variance_all = [acc for model, acc in accuracies_all.items(
    ) if model in model_list and not np.isnan(acc)]
    variance_all = np.var(accuracies_for_variance_all) if len(
        accuracies_for_variance_all) > 1 else np.nan
    print(
        f"\n所有模型在所有题目上的 Kendall's $\tau$: {tau_all:.4f} (p-value={p_all:.4f})")
    print(f"所有模型在所有题目上的方差: {variance_all:.4f}")

    # 领先模型（w/o weak models）
    accuracies_leading = {model: acc for model, acc in accuracies_all.items(
    ) if model in leading_models and not np.isnan(acc)}
    sorted_accuracies_leading = sorted(
        accuracies_leading.items(), key=lambda x: x[1], reverse=True)
    accuracy_ranking_leading = {
        model: rank for rank, (model, _) in enumerate(sorted_accuracies_leading, 1)}
    common_models_leading = [
        m for m in leading_models if m in accuracy_ranking_leading and m in manual_reference_ranking and manual_reference_ranking[m] is not None]
    if len(common_models_leading) < 2:
        print(
            f"警告：w/o weak models 没有足够的共同模型（{len(common_models_leading)}）来计算 Kendall's $\tau$。")
        tau_leading, p_leading = np.nan, np.nan
    else:
        accuracy_ranks_leading = pd.Series(
            accuracy_ranking_leading).reindex(common_models_leading)
        ref_ranks_leading = pd.Series(
            manual_reference_ranking).reindex(common_models_leading)
        tau_leading, p_leading = kendalltau(
            accuracy_ranks_leading, ref_ranks_leading)

    accuracies_for_variance_leading = [
        acc for model, acc in accuracies_leading.items()]
    variance_leading = np.var(accuracies_for_variance_leading) if len(
        accuracies_for_variance_leading) > 1 else np.nan
    print(
        f"领先模型在所有题目上的 Kendall's $\tau$: {tau_leading:.4f} (p-value={p_leading:.4f})")
    print(f"领先模型在所有题目上的方差: {variance_leading:.4f}")

    # 基于 0/1 结果向量聚类的题目选择
    print("\n基于 0/1 结果向量聚类的题目选择结果：")
    num_items_list = [400, 1000]  # 与 Table~\ref{tab:selection} 一致
    for num_items in num_items_list:
        print(f"\n选择 {num_items} 个题目：")
        result = evaluate_clustering_selection(
            model_responses_df, num_items, model_list, leading_models, manual_reference_ranking, random_seed
        )
        print(
            f"w/ weak models: Variance = {result['variance_all']:.4f}, Kendall's $\tau$ = {result['tau_all']:.4f} (p-value={result['p_all']:.4f})")
        print(
            f"w/o weak models: Variance = {result['variance_leading']:.4f}, Kendall's $\tau$ = {result['tau_leading']:.4f} (p-value={result['p_leading']:.4f})")
