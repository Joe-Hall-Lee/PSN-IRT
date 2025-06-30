import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)
# 读取数据
try:
    df = pd.read_excel("opencompass.xlsx")
except FileNotFoundError:
    print("错误: Excel文件 'opencompass.xlsx' 未找到。请确保文件路径正确。")
    exit()


# 定义参数
n = 8
datasets = [
    "HumanEval",
    "BBH",
    "MMLU-Pro",
    "GPQA-Diamond",
]


def wrap_model_name(name, max_width=100):
    return "\n".join(
    textwrap.wrap(
    name, width=max_width, break_long_words=False, break_on_hyphens=True
    )
)


# 计算方差并准备绘图数据
plot_data = {}
variances = {}

for dataset in datasets:
    if dataset not in df.columns:
        print(f"警告: 数据集 '{dataset}' 在 Excel 文件中未找到，将跳过。")
        variances[dataset] = np.nan
        plot_data[dataset] = pd.DataFrame(columns=['模型', dataset])
        continue
    temp_df = df[["模型", dataset]].dropna(subset=[dataset])
    if temp_df.empty or len(temp_df) < n:
        print(f"警告: 数据集 '{dataset}' 在移除缺失值后数据不足 (需要至少 {n} 条)，将跳过。")
        variances[dataset] = np.nan
        plot_data[dataset] = pd.DataFrame(columns=['模型', dataset])
        continue

    sorted_df = temp_df.sort_values(by=dataset, ascending=False).head(n)
    plot_data[dataset] = sorted_df.sort_values(dataset, ascending=True)
    variances[dataset] = sorted_df[dataset].var(ddof=0)


# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

# 绘制每个子图
for idx, dataset in enumerate(datasets):
    if idx >= len(axes):
        break
    
    ax = axes[idx]
    data = plot_data[dataset]

    if data.empty:
        ax.set_title(f"{dataset}\n(No data)", fontsize=plt.rcParams["axes.titlesize"], y=1.02)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    y_pos = np.arange(len(data))
    ax.barh(y_pos, data[dataset], height=0.6, color="#1f77b4", zorder=3)
    wrapped_names = [wrap_model_name(name) for name in data["模型"]]
    ax.barh(y_pos, data[dataset], height=0.6, color="#1f77b4", zorder=3)

    # 设置轴标签和标题
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        wrapped_names,
        ha="right",  # 右对齐
        va="center_baseline",
        fontsize=12,
        weight="bold",
    )
    ax.tick_params(axis="y", left=False, which="both")

    ax.tick_params(axis="x", direction="in", length=4)

    ax.set_xlabel("Score", fontsize=plt.rcParams["axes.labelsize"])
    ax.set_title(
        f"{dataset}\n(Variance: {variances.get(dataset, np.nan):.2f})",
        fontsize=plt.rcParams["axes.titlesize"],
        weight="bold",
        y=1.02
    )

    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    ax.set_xlim(0, 100)

    if not data.empty and len(data[dataset]) > 0:
        plot_range_for_text = 100.0 
        if plot_range_for_text == 0: plot_range_for_text = 1 
        text_offset = plot_range_for_text * 0.015 

        for i, score_val in enumerate(data[dataset]):
            x_pos_text = score_val - text_offset
            text_color = "white"
            horizontal_alignment = "right"

            if x_pos_text < 0 or score_val < (0 + plot_range_for_text * 0.1):
                 x_pos_text = score_val + text_offset
                 text_color = "#1f77b4" 
                 horizontal_alignment = "left"
            
            if horizontal_alignment == "left" and x_pos_text + (len(f"{score_val:.1f}") * 0.5) > 100 :
                 x_pos_text = score_val - text_offset
                 text_color = "white"
                 horizontal_alignment = "right"

            ax.text(
                x_pos_text,
                i,
                f"{score_val:.1f}",
                va="center",
                ha=horizontal_alignment,
                color=text_color,
                fontsize=12, 
                bbox=dict(facecolor="none", edgecolor="none", pad=0.1),
            )

for i in range(len([d for d in datasets if d in plot_data and not plot_data[d].empty]), len(axes)):
    if i < len(axes):
      fig.delaxes(axes[i])

plt.subplots_adjust(
    left=0.08, 
    right=0.95,
    top=0.90, 
    bottom=0.10, 
    hspace=0.40, 
    wspace=0.45, 
)

plt.savefig("intro-fig.png", bbox_inches="tight")
variances_df = pd.DataFrame.from_dict(variances, orient="index", columns=["Variance"])
variances_df = variances_df.dropna()
variances_df.to_csv("dataset_variances.csv")

print(
    "分析完成，图表已保存为 intro-fig.png，方差结果保存为dataset_variances.csv"
)