"""
自定义评估指标实现，覆盖 AUC、Recall、F1、KS 与 Financial Cost。
与论文第五章中定义的指标体系 (式5.18~5.22) 保持一致。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算 Kolmogorov–Smirnov (KS) 统计量，衡量好坏样本累积分布差异。
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def financial_cost(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> float:
    """
    根据式 (5.22) 计算业务财务成本：
        Cost = C_fp * FP + C_fn * FN
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return fp * fp_cost + fn * fn_cost


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    cost_matrix: dict | None = None,
) -> dict:
    """
    汇总模型评估指标。y_score 为连续输出 (违约概率)。
    """
    y_pred = (y_score >= threshold).astype(int)
    metrics = {}

    metrics["auc"] = roc_auc_score(y_true, y_score)
    metrics["recall"] = recall_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["ks"] = ks_statistic(y_true, y_score)

    if cost_matrix:
        fp_cost = cost_matrix.get("false_positive", 1.0)
        fn_cost = cost_matrix.get("false_negative", 1.0)
        metrics["financial_cost"] = financial_cost(y_true, y_pred, fp_cost, fn_cost)
    else:
        metrics["financial_cost"] = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    return metrics
