import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from ..models.atae_gan import Generator, Critic

# ==========================================
# 核心创新点实现：WGAN-GP 训练循环骨架
# ==========================================

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算 WGAN-GP 的梯度惩罚项 (Gradient Penalty)"""
    # 随机权重项 epsilon
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # 在真实样本和生成样本之间进行随机插值
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = D(interpolates)
    
    # 计算判别器输出相对于插值样本的梯度
    fake = torch.ones((real_samples.size(0), 1)).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # 计算梯度的范数 (L2 norm)
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_atae_gan(X_train_minority, latent_dim=128, epochs=200, batch_size=256, n_critic=5, device='cuda'):
    """
    训练 ATAE-GAN 的主循环。
    
    Args:
        X_train_minority (torch.Tensor): 少数类样本数据 (已归一化).
        n_critic (int): 每训练一次生成器，训练判别器的次数 (WGAN推荐为5).
    """
    input_dim = X_train_minority.shape[1]
    
    # 初始化网络
    # 注意：实际应用中，Generator 中的 Decoder 应该先通过 TAE 预训练好
    # 这里为了演示简化了，直接初始化
    from ..models.atae_gan import TabularDecoder 
    decoder = TabularDecoder(latent_dim=latent_dim, output_dim=input_dim)
    generator = Generator(decoder).to(device)
    critic = Critic(input_dim).to(device)
    
    # 优化器 (WGAN 推荐使用 Adam, beta1=0, beta2=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=4e-4, betas=(0.0, 0.9))
    
    # 数据加载器
    dataset = torch.utils.data.TensorDataset(X_train_minority)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    lambda_gp = 10 # 梯度惩罚系数

    print("Starting ATAE-GAN (WGAN-GP) Training...")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, (real_imgs,) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # ===================================
            # 1. 训练 Critic (判别器)
            # ===================================
            optimizer_C.zero_grad()
            
            # 采样噪声并生成假数据
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z).detach() # detach 避免梯度传给生成器
            
            # Critic 对真实和假数据的打分
            real_validity = critic(real_imgs)
            fake_validity = critic(fake_imgs)
            
            # 计算梯度惩罚
            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, device)
            
            # WGAN-GP Loss: D(fake) - D(real) + lambda * gp
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp * gradient_penalty
            
            d_loss.backward()
            optimizer_C.step()
            
            # ===================================
            # 2. 训练 Generator (每 n_critic 步训练一次)
            # ===================================
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                
                # 生成假数据
                z = torch.randn(batch_size, latent_dim).to(device)
                gen_imgs = generator(z)
                
                # Critic 对假数据的打分
                fake_validity = critic(gen_imgs)
                
                # Generator Loss: -D(fake) (希望 Critic 打高分)
                g_loss = -torch.mean(fake_validity)
                
                # (可选创新点) 加入分类器反馈 Loss (Classifier Feedback Loss)
                # classifier_loss = pretrained_classifier(gen_imgs)
                # g_loss = g_loss + lambda_cls * classifier_loss
                
                g_loss.backward()
                optimizer_G.step()
                
        if (epoch + 1) % 20 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            
    print("Training finished.")
    return generator
