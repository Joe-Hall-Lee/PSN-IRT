# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from scipy.stats import kendalltau

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
    if np.isinf(exp_val): # 如果指数结果过大，分母为 inf，icc 趋向于 guessing
        return guessing 
    icc = guessing + (feasibility - guessing) / (1 + exp_val)
    return icc

def calculate_leh_at_ability(difficulty, discrimination, guessing, feasibility, ability=3.7716236):
    """计算在给定能力值处的局部估计 Headroom (LEH) 分数。"""
    icc_at_ability = calculate_icc(ability, difficulty, discrimination, guessing, feasibility)
    icc_slightly_lower = calculate_icc(ability - 0.001, difficulty, discrimination, guessing, feasibility)
    leh = (icc_at_ability - icc_slightly_lower) / 0.001
    return leh, icc_at_ability

def calculate_fisher_information_at_ability(discrimination, p_theta, guessing, feasibility):
    """计算在给定能力值 theta 处的 Fisher 信息。"""
    # 检查 p_theta 是否在 (guessing, feasibility) 的有效范围内
    if p_theta <= guessing or p_theta >= feasibility:
        return 0
    
    # 检查 (feasibility - guessing) 是否为零
    if (feasibility - guessing) == 0:
        return 0
        
    # 检查 p_theta * (1 - p_theta) 是否为零 (这通常意味着 p_theta 是 0 或 1)
    # 如果 p_theta 已经是 guessing 或 feasibility，上面的检查会处理
    # 但如果 guessing=0, feasibility=1, p_theta 可能为 0 或 1
    if p_theta == 0 or p_theta == 1: # 严格等于0或1
        if guessing < p_theta < feasibility: # 这种情况理论上不应该发生，除非参数极端
             pass # 允许计算
        else: # p_theta 在边界上，且等于0或1
             return 0

    # 避免对极小值进行除法或对数操作，增加稳定性
    epsilon = 1e-9 
    if abs(p_theta - guessing) < epsilon or abs(feasibility - p_theta) < epsilon:
        # 如果 p_theta 非常接近 guessing 或 feasibility，分子可能接近 0
        pass

    if p_theta < epsilon or (1 - p_theta) < epsilon: # p_theta 接近 0 或 1 导致分母中的 p_theta*(1-p_theta) 接近0
        if not (p_theta <= guessing or p_theta >= feasibility) : # 确保不是已经被前面条件覆盖的边界情况
            return 0


    numerator = discrimination**2 * (p_theta - guessing)**2 * (feasibility - p_theta)**2
    denominator_factor = p_theta * (1 - p_theta)
    
    if denominator_factor == 0:
        return 0
        
    denominator = (feasibility - guessing)**2 * denominator_factor
    
    if denominator == 0: 
        return 0
            
    return numerator / denominator

def analyze_irt_results_with_ranking(item_file="item_parameters.csv",
                                      ability_for_leh_fisher=3.7716236):
    """计算题目参数的 LEH 和 Fisher 信息，并进行排名（包括排除难度的总排名）。"""
    item_df = pd.read_csv(item_file)

    leh_p_theta = item_df.apply(lambda row: calculate_leh_at_ability(
        row['difficulty'], row['discrimination'], row['guessing'], row['feasibility'], ability=ability_for_leh_fisher
    ), axis=1, result_type='expand')
    item_df['LEH'] = leh_p_theta[0]
    item_df['P_at_ability'] = leh_p_theta[1]

    item_df['Fisher_Info'] = item_df.apply(lambda row: calculate_fisher_information_at_ability(
        row['discrimination'], row['P_at_ability'], row['guessing'], row['feasibility']
    ), axis=1)

    rank_df = item_df.copy()
    rank_df['Difficulty_Rank'] = rank_df['difficulty'].rank(ascending=False) 
    rank_df['Discrimination_Rank'] = rank_df['discrimination'].rank(ascending=False)
    rank_df['Guessing_Rank'] = rank_df['guessing'].rank(ascending=True)      
    rank_df['Feasibility_Rank'] = rank_df['feasibility'].rank(ascending=False)  
    rank_df['LEH_Rank'] = rank_df['LEH'].rank(ascending=False)              
    rank_df['Fisher_Info_Rank'] = rank_df['Fisher_Info'].rank(ascending=False) 

    rank_df['Total_Rank_No_Difficulty'] = (rank_df['Discrimination_Rank'] +
                                           rank_df['Guessing_Rank'] + 
                                           rank_df['Feasibility_Rank'] + 
                                           rank_df['LEH_Rank'] +
                                           rank_df['Fisher_Info_Rank'])
    return rank_df

def select_high_quality_items_by_rank(ranked_item_df, rank_column, num_items=10000, ascending=True):
    """根据指定排名字段选择高质量题目。ascending=True 表示rank数值越小越好。"""
    selected_items = ranked_item_df.sort_values(by=rank_column, ascending=ascending).head(num_items)
    return selected_items.index.tolist()

def evaluate_model_accuracy_on_subset(model_responses_df, item_indices, model_list):
    """评估模型在指定题目子集上的准确率。"""
    model_accuracies = {}
    for i, model in enumerate(model_list):
        if 0 <= i < model_responses_df.shape[0]:
            valid_item_indices = [idx for idx in item_indices if idx < model_responses_df.shape[1]]
            if not valid_item_indices:
                print(f"警告：对于模型 '{model}'，在所选 item_indices 中没有有效的列索引。")
                model_accuracies[model] = np.nan
                continue
            model_responses_subset = model_responses_df.iloc[i, valid_item_indices]
            if model_responses_subset.empty:
                 model_accuracies[model] = np.nan
            else:
                accuracy = model_responses_subset.mean().round(4)
                model_accuracies[model] = accuracy
        else:
            model_accuracies[model] = np.nan
            print(f"警告：模型列表中的模型 '{model}' 在 combine.csv 中没有对应的行 ({i})。")
    return model_accuracies

if __name__ == "__main__":
    try:
        ranked_item_df = analyze_irt_results_with_ranking(item_file="item_parameters.csv")
    except FileNotFoundError:
        print(f"错误: 题目参数文件 'item_parameters.csv' 未找到。无法进行IRT分析。")
        exit()
    except KeyError as e:
        print(f"错误: 'item_parameters.csv' 文件中缺少必要的列: {e}。请确保包含 'difficulty', 'discrimination', 'guessing', 'feasibility'。")
        exit()
    except Exception as e:
        print(f"分析 IRT 结果时发生其他错误: {e}")
        exit()


    model_list = ['360GPT2-Pro', 'DeepSeek-V3', 'Doubao-Pro', 'Gemini-1.5', 'Gemma-2B',
                  'Hunyuan-Turbo', 'Mistral-7B', 'Moonshot-v1', 'Qwen-Plus', 'Qwen2.5-3B',
                  'Vicuna-7B', 'Yi-Lightning']
    open_source_models = ['Mistral-7B', 'Qwen2.5-3B', 'Vicuna-7B', 'Gemma-2B']
    # open_source_models = []

    leading_models = [model for model in model_list if model not in open_source_models]

    manual_reference_ranking = {
        'DeepSeek-V3': 1,
        'Hunyuan-Turbo': 2,
        'Yi-Lightning': 3,
        'Qwen-Plus': 4,
        'Gemini-1.5': 5,
        '360GPT2-Pro': 6,
        'Doubao-Pro': None, # 手动排名为 None
        'Moonshot-v1': 7,
        'Qwen2.5-3B': None, # 手动排名为 None
        'Mistral-7B': 10,
        'Vicuna-7B': 11,
        'Gemma-2B': 12
    }
    
    missing_in_manual_ranking = [m for m in leading_models if m not in manual_reference_ranking]
    if missing_in_manual_ranking:
        print(f"警告：以下领先模型在 'manual_reference_ranking' 中没有定义排名条目: {missing_in_manual_ranking}。这些模型将不会参与相关性计算。")
    
    student_ability_ranking = manual_reference_ranking 

    try:
        model_responses_df = pd.read_csv("combine.csv", header=None)
        num_total_items = model_responses_df.shape[1]
    except FileNotFoundError as e:
        print(f"错误：找不到模型作答文件 {e.filename}，无法进行评估。")
        exit()
    except Exception as e:
        print(f"读取 combine.csv 时发生错误: {e}")
        exit()


    num_high_quality_items = 400
    random_seed = 42 
    random.seed(random_seed)
    np.random.seed(random_seed)

    accuracies_all = {}
    if model_responses_df.shape[0] >= len(model_list):
        if model_responses_df.shape[0] > len(model_list):
            print(f"警告：combine.csv 的行数 ({model_responses_df.shape[0]}) 大于模型列表长度 ({len(model_list)})。仅使用与模型列表对应的前 {len(model_list)} 行。")
        for i, model in enumerate(model_list):
            accuracies_all[model] = model_responses_df.iloc[i, :].mean().round(4)
    else: 
        print(f"错误：combine.csv 的行数 ({model_responses_df.shape[0]}) 小于模型列表长度 ({len(model_list)})。无法为所有模型计算准确率。")
        for i in range(model_responses_df.shape[0]): # 仅为存在的行计算
            if i < len(model_list):
                 accuracies_all[model_list[i]] = model_responses_df.iloc[i, :].mean().round(4)
        print(f"已为 combine.csv 中存在的前 {model_responses_df.shape[0]} 个模型计算准确率。未在 combine.csv 中找到数据的模型将没有准确率。")

    print("\n所有模型在所有题目上的准确率:")
    if accuracies_all:
        sorted_accuracies_all = sorted(accuracies_all.items(), key=lambda item: item[1], reverse=True)
        for rank, (model, accuracy) in enumerate(sorted_accuracies_all, 1):
            print(f"  排名 {rank}: {model} - 准确率: {accuracy}")

        accuracy_ranking_all = {model: rank for rank, (model, _) in enumerate(sorted_accuracies_all, 1)}

        common_models_for_corr_all = [
            m for m in leading_models 
            if m in accuracy_ranking_all and \
               m in student_ability_ranking and \
               student_ability_ranking[m] is not None # 关键条件
        ]
        
        num_common_all = len(common_models_for_corr_all)
        if not common_models_for_corr_all or num_common_all < 2 :
            print(f"\n警告：在 'leading_models' 中没有足够的共同模型 (至少2个，且具有有效手动排名和计算出的准确率) 来进行所有题目的 Kendall Tau 相关系数计算。共同模型数量: {num_common_all}")
            tau_all, p_all = np.nan, np.nan
        else:
            accuracy_ranks_all_series = pd.Series(accuracy_ranking_all).reindex(common_models_for_corr_all)
            irt_ranks_all_series = pd.Series(student_ability_ranking).reindex(common_models_for_corr_all)
            tau_all, p_all = kendalltau(accuracy_ranks_all_series, irt_ranks_all_series)
        
        print(f"\n领先模型在所有题目上准确率排名与手动参考排名的 Kendall Tau 相关系数: {tau_all:.4f} (p-value={p_all:.4f}) (基于 {num_common_all} 个共同模型)")

        # accuracies_all_valid_leading 用于方差计算时，也应该只考虑那些实际参与了相关性计算的模型或所有有准确率的 leading_models
        accuracies_for_variance_calc_all = [
            acc for model, acc in accuracies_all.items() 
            if model in leading_models and not np.isnan(acc) # 确保是 leading_model 且准确率有效
        ]
        variance_all_leading = np.var(accuracies_for_variance_calc_all) if len(accuracies_for_variance_calc_all) > 1 else np.nan
        print(f"领先模型在所有题目上准确率的方差 (基于 {len(accuracies_for_variance_calc_all)} 个模型): {variance_all_leading:.4f}")
    else:
        print("未能计算任何模型的整体准确率。")

    variance_random_leading = np.nan
    tau_random, p_random = np.nan, np.nan
    
    if num_total_items >= num_high_quality_items:
        random_item_indices = random.sample(range(num_total_items), num_high_quality_items)
        model_accuracies_random = evaluate_model_accuracy_on_subset(model_responses_df, random_item_indices, model_list)

        print("\n领先模型在随机选择的题目上的准确率:")
        sorted_accuracies_random_leading_valid = [
            (model, acc) for model, acc in model_accuracies_random.items() 
            if model in leading_models and not np.isnan(acc)
        ]
        sorted_accuracies_random_leading = sorted(
            sorted_accuracies_random_leading_valid,
            key=lambda item: item[1], reverse=True
        )
        for rank, (model, accuracy) in enumerate(sorted_accuracies_random_leading, 1):
            print(f"  排名 {rank}: {model} - 准确率: {accuracy:.4f}")

        accuracy_ranking_random = {model: rank for rank, (model, _) in enumerate(sorted_accuracies_random_leading, 1)}
        
        common_models_for_corr_random = [
            m for m in leading_models 
            if m in accuracy_ranking_random and \
               m in student_ability_ranking and \
               student_ability_ranking[m] is not None # 关键条件
        ]
        num_common_random = len(common_models_for_corr_random)

        if not common_models_for_corr_random or num_common_random < 2:
            print(f"\n警告：在 'leading_models' 中没有足够的共同模型 (至少2个，且具有有效手动排名和计算出的准确率) 来进行随机选择题目的 Kendall Tau 相关系数计算。共同模型数量: {num_common_random}")
            tau_random, p_random = np.nan, np.nan
        else:
            accuracy_ranks_random_series = pd.Series(accuracy_ranking_random).reindex(common_models_for_corr_random)
            irt_ranks_random_series = pd.Series(student_ability_ranking).reindex(common_models_for_corr_random)
            tau_random, p_random = kendalltau(accuracy_ranks_random_series, irt_ranks_random_series)
        
        print(f"\n领先模型在随机选择的题目上准确率排名与手动参考排名的 Kendall Tau 相关系数: {tau_random:.4f} (p-value={p_random:.4f}) (基于 {num_common_random} 个共同模型)")
        
        accuracies_for_variance_calc_random = [acc for model, acc in sorted_accuracies_random_leading_valid]
        variance_random_leading = np.var(accuracies_for_variance_calc_random) if len(accuracies_for_variance_calc_random) > 1 else np.nan
        print(f"领先模型在随机选择的相同数量题目上的准确率方差 (基于 {len(accuracies_for_variance_calc_random)} 个模型): {variance_random_leading:.4f}")
    else:
        print(f"\n警告：题目总数 ({num_total_items}) 小于要随机选择的数量 ({num_high_quality_items})。跳过随机选择部分的计算。")

    metrics_results = {}
    metrics = {
        'Difficulty': ('Difficulty_Rank', True),
        'Discrimination': ('Discrimination_Rank', True),
        'Guessing': ('Guessing_Rank', True),    
        'Feasibility': ('Feasibility_Rank', True),
        'LEH': ('LEH_Rank', True),            
        'Fisher_Info': ('Fisher_Info_Rank', True),
        'Total_No_Difficulty': ('Total_Rank_No_Difficulty', True)
    }

    print("\n基于 IRT 指标选定的高质量题目上领先模型准确率与手动参考排名的 Kendall Tau 相关系数:")
    for metric_name, (rank_column, ascending_sort) in metrics.items():
        if rank_column not in ranked_item_df.columns:
            print(f"警告: 排名列 '{rank_column}' 在 ranked_item_df 中不存在。跳过指标 '{metric_name}'。")
            continue

        selected_item_indices = select_high_quality_items_by_rank(ranked_item_df, rank_column, num_high_quality_items, ascending_sort)
        
        if not selected_item_indices:
            print(f"\n基于 {metric_name} 未选出任何高质量题目。")
            continue

        model_accuracies_hq = evaluate_model_accuracy_on_subset(model_responses_df, selected_item_indices, model_list)

        print(f"\n基于 {metric_name} 选定的高质量题目上领先模型的准确率:")
        sorted_accuracies_hq_leading_valid = [
            (model, acc) for model, acc in model_accuracies_hq.items() 
            if model in leading_models and not np.isnan(acc)
        ]
        sorted_accuracies_hq_leading = sorted(
            sorted_accuracies_hq_leading_valid,
            key=lambda item: item[1], reverse=True
        )

        for rank, (model, accuracy) in enumerate(sorted_accuracies_hq_leading, 1):
            print(f"  排名 {rank}: {model} - 准确率: {accuracy:.4f}")
        
        accuracies_for_variance_calc_hq = [acc for model, acc in sorted_accuracies_hq_leading_valid]
        variance_hq_leading = np.var(accuracies_for_variance_calc_hq) if len(accuracies_for_variance_calc_hq) > 1 else np.nan
        print(f"  方差 (基于 {len(accuracies_for_variance_calc_hq)} 个模型): {variance_hq_leading:.4f}")

        accuracy_ranking_hq = {model: rank for rank, (model, _) in enumerate(sorted_accuracies_hq_leading, 1)}
        
        tau_hq, p_hq = np.nan, np.nan
        
        # MODIFIED: Ensure student_ability_ranking[m] is not None
        common_models_for_corr_hq = [
            m for m in leading_models 
            if m in accuracy_ranking_hq and \
               m in student_ability_ranking and \
               student_ability_ranking[m] is not None # 关键条件
        ]
        num_common_hq = len(common_models_for_corr_hq)

        if not common_models_for_corr_hq or num_common_hq < 2:
            print(f"  警告：在 'leading_models' 中没有足够的共同模型（至少 2 个，且具有有效手动排名和计算出的准确率）来进行 {metric_name} 指标的 Kendall Tau 相关系数计算。共同模型数量: {num_common_hq}")
            tau_hq, p_hq = np.nan, np.nan
        else:
            accuracy_ranks_hq_series = pd.Series(accuracy_ranking_hq).reindex(common_models_for_corr_hq)
            irt_ranks_hq_series = pd.Series(student_ability_ranking).reindex(common_models_for_corr_hq)
            tau_hq, p_hq = kendalltau(accuracy_ranks_hq_series, irt_ranks_hq_series)
        
        print(f"  Kendall Tau 相关系数: {tau_hq:.4f} (p-value={p_hq:.4f}) (基于 {num_common_hq} 个共同模型)")