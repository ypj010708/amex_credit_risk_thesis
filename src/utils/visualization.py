"""
训练指标可视化模块：对第 5 章指标体系 (AUC/Recall/F1/KS/Financial Cost)
生成横向对比图，便于实验结果复现与报告书写。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def _results_to_dataframe(results: dict) -> pd.DataFrame:
    rows: List[dict] = []

    def append_rows(model_name: str, stage: str, metrics: Dict[str, float]):
        for metric_name, value in metrics.items():
            if metric_name == "confusion_matrix":
                continue
            if value is None:
                continue
            rows.append(
                {
                    "model": model_name,
                    "stage": stage,
                    "metric": metric_name,
                    "value": value,
                }
            )

    stacking = results.get("stacking", {})
    for stage, metrics in stacking.items():
        append_rows("Stacking", stage, metrics)

    baselines = results.get("baselines", {})
    for model_name, stages in baselines.items():
        for stage, metrics in stages.items():
            append_rows(model_name, stage, metrics)

    return pd.DataFrame(rows)


def visualize_training_metrics(results: dict, output_dir: str):
    """
    将训练/验证/测试阶段的指标绘制成柱状图并保存到 output_dir。
    """
    df = _results_to_dataframe(results)
    if df.empty:
        print("[Visualization] No metrics to plot.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["auc", "recall", "f1", "ks", "financial_cost"]
    for metric in metrics_to_plot:
        metric_df = df[df["metric"] == metric].copy()
        if metric_df.empty:
            continue

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=metric_df,
            x="model",
            y="value",
            hue="stage",
            palette="viridis" if metric != "financial_cost" else "magma",
        )
        plt.title(f"{metric.upper()} comparison")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        file_suffix = metric.lower()
        plt.savefig(output_path / f"{file_suffix}_comparison.png", dpi=300)
        plt.close()
        print(f"[Visualization] Saved {metric} chart to {output_path}")
