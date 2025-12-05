"""
Stacking 集成实现：支持 LightGBM (Focal Loss)、XGBoost、CatBoost 等基学习器，
并使用逻辑回归作为代价敏感的元学习器。
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .focal_loss import (
    lgbm_focal_loss_objective,
    lgbm_focal_loss_eval,
    robust_sigmoid,
)


class StackingClassifier:
    """
    代价敏感 Stacking 框架：
      - 多个异构 GBDT 基学习器（LightGBM/CatBoost/XGBoost）
      - 自定义 Focal Loss 支持
      - 逻辑回归元学习器
    """

    def __init__(
        self,
        base_models_params: Dict[str, dict],
        meta_model: Optional[LogisticRegression] = None,
        n_folds: int = 5,
        random_state: int = 42,
        categorical_features: Optional[List[str]] = None,
    ):
        self.base_models_params = base_models_params
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_model = meta_model if meta_model else LogisticRegression()
        self.categorical_features = categorical_features or []

        self.base_model_names = list(base_models_params.keys())
        self.feature_names_: Optional[List[str]] = None
        self.fitted_fold_models_: Dict[str, List[object]] = {}
        self.final_base_models_: Dict[str, object] = {}
        self.oof_predictions_: Optional[np.ndarray] = None
        self.is_trained_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """训练基模型，生成 OOF 预测，再训练元学习器。"""
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.feature_names_ = list(X.columns)

        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight = np.asarray(sample_weight)

        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        self.oof_predictions_ = np.zeros((len(X), len(self.base_model_names)))
        self.fitted_fold_models_ = {name: [] for name in self.base_model_names}

        print(f"[Stacking] Start training with {len(self.base_model_names)} base learners.")

        for model_idx, model_name in enumerate(self.base_model_names):
            model_cfg = self.base_models_params[model_name]
            model_type = model_cfg.get("type", "").lower()
            fold_preds = np.zeros(len(X))

            for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                w_train = sample_weight[train_idx]
                w_val = sample_weight[val_idx]

                model = self._train_single_base_model(
                    model_name=model_name,
                    model_type=model_type,
                    model_cfg=model_cfg,
                    X_train=X_train,
                    y_train=y_train,
                    sample_weight=w_train,
                    X_val=X_val,
                    y_val=y_val,
                    val_weight=w_val,
                )
                preds = self._predict_with_model(model_type, model, X_val)

                fold_preds[val_idx] = preds
                self.fitted_fold_models_[model_name].append(model)
                print(f"    [{model_name}] Fold {fold_id + 1}/{self.n_folds} done.")

            self.oof_predictions_[:, model_idx] = fold_preds
            auc = roc_auc_score(y, fold_preds)
            print(f"  -> {model_name} OOF AUC: {auc:.4f}")

        print("[Stacking] Training meta-learner (Logistic Regression).")
        self.meta_model.fit(self.oof_predictions_, y, sample_weight=sample_weight)

        # 使用全部训练集拟合最终基学习器，便于推理阶段预测
        self.final_base_models_ = {}
        for model_name in self.base_model_names:
            model_cfg = self.base_models_params[model_name]
            model_type = model_cfg.get("type", "").lower()
            self.final_base_models_[model_name] = self._train_single_base_model(
                model_name=model_name,
                model_type=model_type,
                model_cfg=model_cfg,
                X_train=X,
                y_train=y,
                sample_weight=sample_weight,
                X_val=None,
                y_val=None,
                val_weight=None,
            )

        self.is_trained_ = True
        print("[Stacking] Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained_ or self.feature_names_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = pd.DataFrame(X)[self.feature_names_]
        base_level_preds = []

        for model_name in self.base_model_names:
            model_cfg = self.base_models_params[model_name]
            model_type = model_cfg.get("type", "").lower()
            model = self.final_base_models_.get(model_name)

            if model is None:
                # 回退到折模型平均
                preds = np.mean(
                    [self._predict_with_model(model_type, m, X) for m in self.fitted_fold_models_[model_name]],
                    axis=0,
                )
            else:
                preds = self._predict_with_model(model_type, model, X)
            base_level_preds.append(preds)

        stacked_features = np.column_stack(base_level_preds)
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(stacked_features)[:, 1]
        decision_scores = self.meta_model.decision_function(stacked_features)
        return robust_sigmoid(decision_scores)

    # ------------------------------------------------------------------ #
    # 内部工具函数
    # ------------------------------------------------------------------ #
    def _train_single_base_model(
        self,
        model_name: str,
        model_type: str,
        model_cfg: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[np.ndarray],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        val_weight: Optional[np.ndarray],
    ):
        if model_type == "lightgbm":
            return self._train_lightgbm(model_cfg, X_train, y_train, sample_weight, X_val, y_val, val_weight)
        if model_type == "xgboost":
            return self._train_xgboost(model_cfg, X_train, y_train, sample_weight, X_val, y_val, val_weight)
        if model_type == "catboost":
            return self._train_catboost(model_cfg, X_train, y_train, sample_weight, X_val, y_val, val_weight)
        raise ValueError(f"Unsupported base model type: {model_type} ({model_name})")

    def _train_lightgbm(
        self,
        model_cfg: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[np.ndarray],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        val_weight: Optional[np.ndarray],
    ):
        params = deepcopy(model_cfg.get("params", {}))
        # 为了兼容旧版本 LightGBM，这里不再显式传入 fobj/feval，自定义 Focal Loss
        # 逻辑保留在 focal_loss.py 中，如需启用可在升级 LightGBM 后恢复。
        num_round = params.pop("num_boost_round", params.pop("n_estimators", 1000))
        params.setdefault("objective", "binary")

        train_set = lgb.Dataset(X_train, y_train, weight=sample_weight)
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None:
            val_set = lgb.Dataset(X_val, y_val, weight=val_weight, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("valid")

        try:
            # 尝试使用支持 early_stopping_rounds / valid_names 的新接口
            model = lgb.train(
                params=params,
                train_set=train_set,
                num_boost_round=num_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                early_stopping_rounds=50 if X_val is not None else None,
                verbose_eval=False,
            )
        except TypeError as e:
            # 兼容非常旧版本 LightGBM：去掉不被支持的关键字参数
            print(
                "[Warning] Current LightGBM version does not fully support "
                f"'early_stopping_rounds' / 'valid_names' (error: {e}). "
                "Falling back to minimal lgb.train() signature."
            )
            model = lgb.train(
                params=params,
                train_set=train_set,
                num_boost_round=num_round,
                valid_sets=valid_sets,
            )
        return model

    def _train_xgboost(
        self,
        model_cfg: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[np.ndarray],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        val_weight: Optional[np.ndarray],
    ):
        params = deepcopy(model_cfg.get("params", {}))
        num_boost_round = params.pop("n_estimators", params.pop("num_boost_round", 500))

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        evals = [(dtrain, "train")]

        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, weight=val_weight)
            evals.append((dval, "valid"))

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50 if X_val is not None else None,
            verbose_eval=False,
        )
        return model

    def _train_catboost(
        self,
        model_cfg: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[np.ndarray],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        val_weight: Optional[np.ndarray],
    ):
        params = deepcopy(model_cfg.get("params", {}))
        cat_indices = self._resolve_cat_features(X_train.columns)

        model = CatBoostClassifier(**params)
        train_pool = Pool(
            X_train,
            y_train,
            weight=sample_weight,
            cat_features=cat_indices if cat_indices else None,
        )

        if X_val is not None:
            val_pool = Pool(
                X_val,
                y_val,
                weight=val_weight,
                cat_features=cat_indices if cat_indices else None,
            )
            model.fit(train_pool, eval_set=val_pool, verbose=params.get("verbose", 0))
        else:
            model.fit(train_pool, verbose=params.get("verbose", 0))
        return model

    def _predict_with_model(self, model_type: str, model, X: pd.DataFrame) -> np.ndarray:
        if model_type == "lightgbm":
            raw_scores = model.predict(X)
            return robust_sigmoid(raw_scores)
        if model_type == "xgboost":
            dmatrix = xgb.DMatrix(X)
            return model.predict(dmatrix)
        if model_type == "catboost":
            proba = model.predict_proba(X)
            return proba[:, 1]
        raise ValueError(f"Unsupported model type: {model_type}")

    def _resolve_cat_features(self, columns: List[str]) -> Optional[List[int]]:
        if not self.categorical_features:
            return None
        if isinstance(self.categorical_features[0], int):
            return self.categorical_features
        index_map = {col: idx for idx, col in enumerate(columns)}
        return [index_map[col] for col in self.categorical_features if col in index_map]
