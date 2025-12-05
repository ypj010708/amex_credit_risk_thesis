import numpy as np
from scipy.misc import derivative

# ==========================================
# 核心创新点实现：GBDT (LightGBM/XGBoost) 自定义 Focal Loss
# ==========================================

def robust_sigmoid(x):
    """数值稳定的 Sigmoid 函数，防止溢出"""
    # 对大于0和小于0的部分分别处理
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def lgbm_focal_loss_objective(preds, train_data, alpha=0.25, gamma=2.0):
    """
    LightGBM 的自定义目标函数 (Objective Function) 实现 Focal Loss。
    需要计算一阶导数 (Gradient) 和二阶导数 (Hessian)。
    
    Focal Loss = -alpha * (1-pt)^gamma * log(pt)
    其中 pt 是模型预测正类的概率。
    
    Args:
        preds: 模型的原始输出 (raw scores / log(odds)), 需要做 sigmoid 转换。
        train_data: LightGBM 的 Dataset 对象，包含真实标签 y_true。
        alpha (float): 平衡正负样本权重的参数 (0 < alpha < 1)。alpha 用于正样本。
        gamma (float): 聚焦参数 (gamma >= 0)，控制对难分样本的关注度。
        
    Returns:
        grad: 一阶导数
        hess: 二阶导数
    """
    y_true = train_data.get_label()
    # 将原始分数转换为概率
    p = robust_sigmoid(preds)
    
    # 为了数值稳定性，将概率限制在 (epsilon, 1-epsilon) 之间
    epsilon = 1e-9
    p = np.clip(p, epsilon, 1. - epsilon)
    
    # --- 核心公式推导实现 ---
    # pt: 模型预测正确类别的概率
    # 如果 y=1, pt = p; 如果 y=0, pt = 1-p
    pt = np.where(y_true == 1, p, 1 - p)
    
    # alpha_t: 类别权重
    # 如果 y=1, alpha_t = alpha; 如果 y=0, alpha_t = 1 - alpha
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    
    # 聚焦项 (modulating factor)
    modulating_factor = (1 - pt) ** gamma
    
    # --- 计算一阶导数 (Gradient) ---
    # d(FL)/d(score) 的推导较为复杂，这里使用其解析解形式
    # grad = alpha_t * (1-pt)^gamma * (gamma * pt * log(pt) + pt - 1) * (如果是y=0则乘以-1)
    # 下面是针对 sigmoid 输出化简后的梯度公式
    
    # 通用梯度项
    grad_common = alpha_t * modulating_factor
    # 针对 y=1 和 y=0 的特定项
    grad_y1 = grad_common * (gamma * (1 - p) * np.log(p) + (p - 1))
    grad_y0 = grad_common * (gamma * p * np.log(1 - p) + p)
    
    grad = np.where(y_true == 1, grad_y1, grad_y0)

    # --- 计算二阶导数 (Hessian) ---
    # Hessian 的解析解非常复杂且容易出错。
    # 在实践中，对于复杂的自定义 Loss，常常使用一个常数或者简化的近似值。
    # 一个常见的近似是用二元交叉熵的 Hessian: p * (1-p)
    # 或者使用更精确但计算量大的近似。
    # 这里采用一种稳健的近似方案：
    
    hess = p * (1 - p) # 基础 BCE Hessian
    # 乘以一个基于 Focal Loss 的调整系数，确保这一项主要是正的
    hess_adjust = alpha_t * modulating_factor * (1 + gamma * (1-pt) + gamma*(gamma-1)*(1-pt)*np.log(pt))
    # 取绝对值确保 Hessian 为正 (凸优化要求)，并添加小量防止除零
    hess = np.abs(hess * hess_adjust) + epsilon

    return grad, hess

def lgbm_focal_loss_eval(preds, train_data, alpha=0.25, gamma=2.0):
    """
    (可选) 用于 LightGBM 评估的 Focal Loss 函数。
    返回 Loss 值本身，用于早停 (early stopping)。
    """
    y_true = train_data.get_label()
    p = robust_sigmoid(preds)
    epsilon = 1e-9
    p = np.clip(p, epsilon, 1. - epsilon)
    
    pt = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    
    # Focal Loss 公式
    loss = -alpha_t * ((1 - pt) ** gamma) * np.log(pt)
    
    return 'focal_loss', np.mean(loss), False # False 表示指标越小越好
