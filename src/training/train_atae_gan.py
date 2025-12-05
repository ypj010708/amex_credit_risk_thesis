"""
ATAE-GAN 训练脚本
=================

该模块实现第四章提出的 Attention-based Tabular AutoEncoder GAN (ATAE-GAN) 训练流程：

1. 数据编码与预处理（标准化、缺失值处理、One-Hot 编码、PCA 可选）
2. ATAE 自编码器阶段性预训练
3. 基于 WGAN-GP + 分类器反馈的两阶段对抗训练
4. 生成合成样本的实用接口

使用示例::

    from src.training.train_atae_gan import (
        ATAEGANConfig, ATAEGANTrainer, preprocess_tabular_dataframe
    )
    tensor, pipeline = preprocess_tabular_dataframe(df, numeric_cols, categorical_cols)
    trainer = ATAEGANTrainer(ATAEGANConfig(input_dim=tensor.shape[1]))
    trainer.fit(minority_tensor=tensor[y==1], full_tensor=tensor, labels=torch.tensor(y))
    synthetic = trainer.sample(1024)
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from ..models.atae_gan import (
    AttentionTabularAutoEncoder,
    ATAEGenerator,
    BoundaryClassifier,
    Critic,
)


# ---------------------------------------------------------------------------
# 数据预处理
# ---------------------------------------------------------------------------

def build_preprocess_pipeline(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    scale_mode: str = "standard",
    use_pca: bool = False,
    pca_components: Optional[int] = None,
) -> Pipeline:
    """
    构建数据预处理流水线：
        - 数值特征：缺失值填充 + 标准化/归一化
        - 类别特征：缺失值填充 + OneHot
        - 可选 PCA 降维
    """
    if scale_mode not in {"standard", "minmax"}:
        raise ValueError("scale_mode must be 'standard' or 'minmax'")

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                StandardScaler() if scale_mode == "standard" else MinMaxScaler(),
            ),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_transformer, list(numeric_cols)))
    if categorical_cols:
        transformers.append(("cat", cat_transformer, list(categorical_cols)))

    column_transformer = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.0
    )

    steps = [("columns", column_transformer)]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, whiten=True)))

    return Pipeline(steps=steps)


def preprocess_tabular_dataframe(
    df,
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    scale_mode: str = "standard",
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    pipeline: Optional[Pipeline] = None,
    fit: bool = True,
) -> Tuple[torch.Tensor, Pipeline]:
    """
    对 DataFrame 进行预处理并返回 torch.Tensor。
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = [
            c for c in df.columns if c not in numeric_cols
        ]

    if pipeline is None:
        pipeline = build_preprocess_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            scale_mode=scale_mode,
            use_pca=use_pca,
            pca_components=pca_components,
        )

    if fit:
        array = pipeline.fit_transform(df)
    else:
        array = pipeline.transform(df)

    if hasattr(array, "toarray"):
        array = array.toarray()
    array = array.astype(np.float32)
    tensor = torch.from_numpy(array)
    return tensor, pipeline


# ---------------------------------------------------------------------------
# 配置与训练器
# ---------------------------------------------------------------------------

@dataclass
class ATAEGANConfig:
    input_dim: int
    latent_dim: int = 128
    noise_dim: int = 128
    attn_embed_dim: int = 64
    attn_heads: int = 4
    attn_seq_len: int = 8

    batch_size: int = 256
    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-3
    adv_epochs: int = 200
    critic_steps: int = 5
    lr_generator: float = 1e-4
    lr_critic: float = 4e-4
    lr_classifier: float = 1e-3

    lambda_gp: float = 10.0
    lambda_classifier: float = 1.0
    enable_classifier_feedback: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """WGAN-GP 梯度惩罚"""
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    critic_scores = critic(interpolates)
    ones = torch.ones_like(critic_scores, device=device)
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


class ATAEGANTrainer:
    """
    将第四章提出的多阶段训练策略封装为可复用 Trainer。
    """

    def __init__(self, config: ATAEGANConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.autoencoder = AttentionTabularAutoEncoder(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            attn_embed_dim=config.attn_embed_dim,
            num_heads=config.attn_heads,
            seq_len_for_attn=config.attn_seq_len,
        ).to(self.device)

        self.generator: Optional[ATAEGenerator] = None
        self.critic = Critic(config.input_dim).to(self.device)
        self.classifier = BoundaryClassifier(config.input_dim).to(self.device)

        self.generator_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.classifier_optimizer: Optional[torch.optim.Optimizer] = None
        
        # 训练历史记录
        self.history = {
            'pretrain_loss': [],
            'critic_loss': [],
            'generator_loss': [],
            'classifier_loss': [],
        }

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def fit(
        self,
        minority_tensor: torch.Tensor,
        full_tensor: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        完整训练流程：
            1) 自编码器预训练
            2) 分类器（可选）训练
            3) WGAN-GP + 分类器反馈的对抗训练
        """
        minority_tensor = minority_tensor.to(self.device)
        self._pretrain_autoencoder(minority_tensor)

        # 初始化生成器，载入解码器参数
        decoder = self.autoencoder.decoder
        self.generator = ATAEGenerator(
            decoder=decoder, noise_dim=self.cfg.noise_dim, latent_dim=self.cfg.latent_dim
        ).to(self.device)
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.cfg.lr_generator, betas=(0.0, 0.9)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr_critic, betas=(0.0, 0.9)
        )

        # 分类器预训练
        if (
            self.cfg.enable_classifier_feedback
            and full_tensor is not None
            and labels is not None
        ):
            self._train_boundary_classifier(full_tensor.to(self.device), labels.to(self.device))
        else:
            self.cfg.enable_classifier_feedback = False  # 明确禁用，避免后续判断

        self._adversarial_training(minority_tensor)

    def sample(self, n_samples: int) -> torch.Tensor:
        if self.generator is None:
            raise RuntimeError("Generator is not trained yet. Call `fit` first.")
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.cfg.noise_dim, device=self.device)
            samples = self.generator(noise)
        return samples.cpu()

    # ------------------------------------------------------------------
    # 阶段 1：ATAE 重构预训练
    # ------------------------------------------------------------------
    def _pretrain_autoencoder(self, tensor: torch.Tensor):
        dataloader = self._build_loader(tensor, shuffle=True)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.cfg.pretrain_lr)
        criterion = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(self.cfg.pretrain_epochs):
            epoch_loss = 0.0
            for batch, in dataloader:
                batch = batch.to(self.device)
                recon, _ = self.autoencoder(batch)
                loss = criterion(recon, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(dataloader)
            self.history['pretrain_loss'].append(avg)
            if (epoch + 1) % max(self.cfg.pretrain_epochs // 5, 1) == 0:
                print(f"[Pretrain {epoch+1}/{self.cfg.pretrain_epochs}] recon_loss={avg:.6f}")

    # ------------------------------------------------------------------
    # 阶段 2A：分类器反馈
    # ------------------------------------------------------------------
    def _train_boundary_classifier(self, features: torch.Tensor, labels: torch.Tensor, epochs: int = 20):
        dataset = TensorDataset(features, labels.float())
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=self.cfg.lr_classifier, weight_decay=1e-4
        )
        criterion = nn.BCEWithLogitsLoss()
        self.classifier.train()
        for epoch in range(epochs):
            loss_epoch = 0.0
            for feat, target in loader:
                feat = feat.to(self.device)
                target = target.to(self.device)
                logits = self.classifier(feat).squeeze(1)
                loss = criterion(logits, target)

                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
                loss_epoch += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"[Classifier {epoch+1}/{epochs}] loss={loss_epoch/len(loader):.4f}")

        self.classifier.eval()

    # ------------------------------------------------------------------
    # 阶段 2B：WGAN-GP + 分类器反馈的对抗训练
    # ------------------------------------------------------------------
    def _adversarial_training(self, minority_tensor: torch.Tensor):
        dataloader = self._build_loader(minority_tensor, shuffle=True, drop_last=True)
        self.generator.train()
        self.critic.train()

        for epoch in range(self.cfg.adv_epochs):
            epoch_critic_loss = 0.0
            epoch_gen_loss = 0.0
            epoch_cls_loss = 0.0
            n_critic_steps = 0
            n_gen_steps = 0
            
            for i, (real_batch,) in enumerate(dataloader):
                real_batch = real_batch.to(self.device)
                batch_size = real_batch.size(0)

                # ----------- Train Critic -----------
                self.critic_optimizer.zero_grad()
                noise = torch.randn(batch_size, self.cfg.noise_dim, device=self.device)
                fake_batch = self.generator(noise).detach()
                real_score = self.critic(real_batch)
                fake_score = self.critic(fake_batch)
                gp = compute_gradient_penalty(self.critic, real_batch, fake_batch, self.device)
                critic_loss = torch.mean(fake_score) - torch.mean(real_score) + self.cfg.lambda_gp * gp
                critic_loss.backward()
                self.critic_optimizer.step()
                epoch_critic_loss += critic_loss.item()
                n_critic_steps += 1

                # ----------- Train Generator -----------
                if i % self.cfg.critic_steps == 0:
                    self.generator_optimizer.zero_grad()
                    noise = torch.randn(batch_size, self.cfg.noise_dim, device=self.device)
                    generated = self.generator(noise)
                    gen_score = self.critic(generated)
                    gen_loss = -torch.mean(gen_score)
                    cls_loss_val = 0.0

                    if self.cfg.enable_classifier_feedback:
                        with torch.no_grad():
                            real_ref = self.classifier(real_batch).detach()
                        fake_logits = self.classifier(generated)
                        cls_loss = F.mse_loss(fake_logits, real_ref)
                        gen_loss = gen_loss + self.cfg.lambda_classifier * cls_loss
                        cls_loss_val = cls_loss.item()

                    gen_loss.backward()
                    self.generator_optimizer.step()
                    epoch_gen_loss += gen_loss.item()
                    epoch_cls_loss += cls_loss_val
                    n_gen_steps += 1
            
            # 记录每个epoch的平均损失
            if n_critic_steps > 0:
                self.history['critic_loss'].append(epoch_critic_loss / n_critic_steps)
            if n_gen_steps > 0:
                self.history['generator_loss'].append(epoch_gen_loss / n_gen_steps)
                if self.cfg.enable_classifier_feedback:
                    self.history['classifier_loss'].append(epoch_cls_loss / n_gen_steps)

            if (epoch + 1) % max(self.cfg.adv_epochs // 10, 1) == 0:
                avg_critic = epoch_critic_loss / n_critic_steps if n_critic_steps > 0 else 0
                avg_gen = epoch_gen_loss / n_gen_steps if n_gen_steps > 0 else 0
                print(
                    f"[Adv {epoch+1}/{self.cfg.adv_epochs}] "
                    f"D_loss={avg_critic:.4f} "
                    f"G_loss={avg_gen:.4f}"
                )

    # ------------------------------------------------------------------
    def _build_loader(self, tensor: torch.Tensor, shuffle: bool, drop_last: bool = False):
        dataset = TensorDataset(tensor)
        return DataLoader(
            dataset,
            batch_size=min(self.cfg.batch_size, len(dataset)),
            shuffle=shuffle,
            drop_last=drop_last,
        )

