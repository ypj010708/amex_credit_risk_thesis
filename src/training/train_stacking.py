"""
基于代价敏感集成学习 (第5章) 的模型训练脚本：
  - 构建 XGBoost / CatBoost / LightGBM (Focal Loss) 异构基学习器
  - 训练带 Logistic Regression 元学习器的 Stacking 框架
  - 训练并评估 LR/SVM/RF/XGBoost/SMOTE-XGBoost 等对比模型
  - 输出 AUC、Recall、F1、KS 与 Financial Cost 指标
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

import sys

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stacking import StackingClassifier
from src.utils.metrics import calculate_classification_metrics
from src.utils.visualization import visualize_training_metrics


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_file(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    if file_path.suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported data format: {file_path.suffix}")


def load_training_data_from_config(config: dict) -> pd.DataFrame:
    """
    加载特征与标签数据：
      - 特征来自 feature_engineering 生成的 parquet
      - 标签来自 data/raw/train_labels.csv
      - 通过 customer_ID 进行关联
    """
    data_cfg = config["data"]
    features_path = data_cfg.get("features_path")
    labels_path = data_cfg.get("labels_path")

    if features_path is None:
        # 回退到单一表形式
        return _load_file(data_cfg["final_train_path"])

    features = _load_file(features_path)
    if labels_path is None:
        return features

    labels = _load_file(labels_path)
    if "customer_ID" not in features.columns or "customer_ID" not in labels.columns:
        raise KeyError("期望特征与标签表均包含 'customer_ID' 列以完成关联。")

    if "target" not in labels.columns:
        raise KeyError("标签表中未找到 'target' 列，请检查 train_labels.csv。")

    df = features.merge(labels[["customer_ID", "target"]], on="customer_ID", how="inner")
    if df["target"].isna().any():
        df = df[~df["target"].isna()].reset_index(drop=True)
    return df


def prepare_splits(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list,
    split_cfg: dict,
    random_state: int,
) -> Dict[str, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in exclude_cols + [target_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    test_ratio = split_cfg["test_ratio"]
    val_ratio = split_cfg["val_ratio"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=adjusted_val_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    return {
        "X_train": X_train.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


def compute_sample_weights(y: pd.Series, cost_weights: dict) -> np.ndarray:
    pos_w = cost_weights.get("positive", 1.0)
    neg_w = cost_weights.get("negative", 1.0)
    return np.where(y.values == 1, pos_w, neg_w)


def run_stacking_training(
    splits: Dict[str, pd.DataFrame],
    config: dict,
    model_cfg: dict,
) -> Dict[str, dict]:
    stacking_cfg = model_cfg["stacking"]
    meta_params = stacking_cfg.get("meta_model", {})
    meta_model = LogisticRegression(**meta_params)

    eval_cfg = config["training"]["evaluation"]
    threshold = eval_cfg["threshold"]
    cost_matrix = eval_cfg["cost_matrix"]
    cost_weights = config["training"]["cost_weights"]
    categorical_features = config["training"].get("categorical_features", [])

    w_train = compute_sample_weights(splits["y_train"], cost_weights)
    w_train_val = compute_sample_weights(splits["y_train_val"], cost_weights)

    stacking_model = StackingClassifier(
        base_models_params=stacking_cfg["base_models"],
        meta_model=meta_model,
        n_folds=config["training"]["n_folds"],
        random_state=config["training"]["random_seed"],
        categorical_features=categorical_features,
    )
    stacking_model.fit(splits["X_train"], splits["y_train"], sample_weight=w_train)
    val_pred = stacking_model.predict_proba(splits["X_val"])
    val_metrics = calculate_classification_metrics(
        splits["y_val"].values,
        val_pred,
        threshold=threshold,
        cost_matrix=cost_matrix,
    )

    # 使用训练集 + 验证集重训，用于最终测试评估
    stacking_full = StackingClassifier(
        base_models_params=stacking_cfg["base_models"],
        meta_model=meta_model.__class__(**meta_model.get_params()),
        n_folds=config["training"]["n_folds"],
        random_state=config["training"]["random_seed"],
        categorical_features=categorical_features,
    )
    stacking_full.fit(splits["X_train_val"], splits["y_train_val"], sample_weight=w_train_val)
    test_pred = stacking_full.predict_proba(splits["X_test"])
    test_metrics = calculate_classification_metrics(
        splits["y_test"].values,
        test_pred,
        threshold=threshold,
        cost_matrix=cost_matrix,
    )

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }


def _prepare_pipeline(
    estimator,
    step_name: str,
    use_scaler: bool,
) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append((step_name, estimator))
    return Pipeline(steps)


def _fit_pipeline(
    pipeline: Pipeline,
    step_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None,
):
    fit_params = {}
    if sample_weight is not None:
        fit_params[f"{step_name}__sample_weight"] = sample_weight
    pipeline.fit(X, y, **fit_params)
    return pipeline


def _predict_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    decision = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
    return decision


def _train_pipeline_model(
    estimator_builder: Callable[[], object],
    step_name: str,
    use_scaler: bool,
    splits: Dict[str, pd.DataFrame],
    threshold: float,
    cost_matrix: dict,
    w_train: np.ndarray | None,
    w_train_val: np.ndarray | None,
) -> Dict[str, dict]:
    # 验证集
    pipeline_val = _prepare_pipeline(estimator_builder(), step_name, use_scaler)
    pipeline_val = _fit_pipeline(
        pipeline_val,
        step_name,
        splits["X_train"],
        splits["y_train"],
        w_train,
    )
    val_pred = _predict_probabilities(pipeline_val, splits["X_val"])
    val_metrics = calculate_classification_metrics(
        splits["y_val"].values,
        val_pred,
        threshold=threshold,
        cost_matrix=cost_matrix,
    )

    # 测试集 (训练 + 验证重新训练)
    pipeline_test = _prepare_pipeline(estimator_builder(), step_name, use_scaler)
    pipeline_test = _fit_pipeline(
        pipeline_test,
        step_name,
        splits["X_train_val"],
        splits["y_train_val"],
        w_train_val,
    )
    test_pred = _predict_probabilities(pipeline_test, splits["X_test"])
    test_metrics = calculate_classification_metrics(
        splits["y_test"].values,
        test_pred,
        threshold=threshold,
        cost_matrix=cost_matrix,
    )

    return {"validation": val_metrics, "test": test_metrics}


def run_baseline_models(
    splits: Dict[str, pd.DataFrame],
    config: dict,
    model_cfg: dict,
) -> Dict[str, dict]:
    eval_cfg = config["training"]["evaluation"]
    threshold = eval_cfg["threshold"]
    cost_matrix = eval_cfg["cost_matrix"]
    cost_weights = config["training"]["cost_weights"]
    w_train = compute_sample_weights(splits["y_train"], cost_weights)
    w_train_val = compute_sample_weights(splits["y_train_val"], cost_weights)

    baselines = model_cfg.get("baselines", {})
    results = {}

    if "logistic_regression" in baselines:
        lr_params = baselines["logistic_regression"]
        builder = lambda: LogisticRegression(**lr_params)
        results["logistic_regression"] = _train_pipeline_model(
            builder,
            step_name="logreg",
            use_scaler=True,
            splits=splits,
            threshold=threshold,
            cost_matrix=cost_matrix,
            w_train=w_train,
            w_train_val=w_train_val,
        )

    if "svm_rbf" in baselines:
        svm_params = baselines["svm_rbf"]
        builder = lambda: SVC(**svm_params)
        results["svm_rbf"] = _train_pipeline_model(
            builder,
            step_name="svc",
            use_scaler=True,
            splits=splits,
            threshold=threshold,
            cost_matrix=cost_matrix,
            w_train=w_train,
            w_train_val=w_train_val,
        )

    if "random_forest" in baselines:
        rf_params = baselines["random_forest"]
        builder = lambda: RandomForestClassifier(**rf_params)
        results["random_forest"] = _train_pipeline_model(
            builder,
            step_name="rf",
            use_scaler=False,
            splits=splits,
            threshold=threshold,
            cost_matrix=cost_matrix,
            w_train=w_train,
            w_train_val=w_train_val,
        )

    if "xgboost" in baselines:
        xgb_params = baselines["xgboost"]
        builder = lambda: XGBClassifier(
            use_label_encoder=False,
            eval_metric=xgb_params.get("eval_metric", "auc"),
            **{k: v for k, v in xgb_params.items() if k != "eval_metric"},
        )
        results["xgboost"] = _train_pipeline_model(
            builder,
            step_name="xgb",
            use_scaler=False,
            splits=splits,
            threshold=threshold,
            cost_matrix=cost_matrix,
            w_train=w_train,
            w_train_val=w_train_val,
        )

    if "smote_xgboost" in baselines:
        smote_cfg = baselines["smote_xgboost"]
        results["smote_xgboost"] = _train_smote_xgb(
            splits,
            smote_cfg,
            threshold=threshold,
            cost_matrix=cost_matrix,
            random_state=config["training"]["random_seed"],
        )

    return results


def _train_smote_xgb(
    splits: Dict[str, pd.DataFrame],
    cfg: dict,
    threshold: float,
    cost_matrix: dict,
    random_state: int,
) -> Dict[str, dict]:
    smote_params = cfg.get("smote", {})
    xgb_params = cfg.get("xgb_params", {})
    base_params = {
        k: v
        for k, v in xgb_params.items()
        if k not in {"eval_metric", "use_label_encoder"}
    }
    eval_metric = xgb_params.get("eval_metric", "auc")

    smote = SMOTE(random_state=random_state, **smote_params)

    # 验证集训练
    X_resampled, y_resampled = smote.fit_resample(splits["X_train"], splits["y_train"])
    model_val = XGBClassifier(
        use_label_encoder=False,
        eval_metric=eval_metric,
        **base_params,
    )
    model_val.fit(X_resampled, y_resampled)
    val_pred = model_val.predict_proba(splits["X_val"])[:, 1]
    val_metrics = calculate_classification_metrics(
        splits["y_val"].values,
        val_pred,
        threshold=threshold,
        cost_matrix=cost_matrix,
    )

    # 测试集训练
    X_full_res, y_full_res = smote.fit_resample(splits["X_train_val"], splits["y_train_val"])
    model_test = XGBClassifier(
        use_label_encoder=False,
        eval_metric=eval_metric,
        **base_params,
    )
    model_test.fit(X_full_res, y_full_res)
    test_pred = model_test.predict_proba(splits["X_test"])[:, 1]
    test_metrics = calculate_classification_metrics(
        splits["y_test"].values,
        test_pred,
        threshold=threshold,
        cost_matrix=cost_matrix,
    )

    return {"validation": val_metrics, "test": test_metrics}


def serialize_results(results: dict, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=default)
    print(f"[Result] Metrics saved to {path}")


def main(config_path: str, model_config_path: str, output_path: str):
    config = load_yaml(config_path)
    model_cfg = load_yaml(model_config_path)

    df = load_training_data_from_config(config)
    splits = prepare_splits(
        df,
        target_col=config["training"]["target_col"],
        exclude_cols=config["training"].get("feature_exclude", []),
        split_cfg=config["training"]["split"],
        random_state=config["training"]["random_seed"],
    )

    stacking_results = run_stacking_training(splits, config, model_cfg)
    baseline_results = run_baseline_models(splits, config, model_cfg)

    all_results = {
        "stacking": stacking_results,
        "baselines": baseline_results,
    }
    serialize_results(all_results, output_path)
    visualize_training_metrics(
        all_results,
        output_dir="results/figures/stacking",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost-sensitive stacking trainer")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-config", default="config/model_params.yaml")
    parser.add_argument(
        "--output",
        default="results/logs/cost_sensitive_stacking.json",
    )
    args = parser.parse_args()
    main(args.config, args.model_config, args.output)
