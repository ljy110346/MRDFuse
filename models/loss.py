import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        probs = F.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1]).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(p_t + 1e-8)
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class JointLoss(nn.Module):
    def __init__(self,
                 use_focal=True,
                 use_mse=True,
                 adaptive_weight=False,
                 lambda_recon=0.5,
                 lambda_class=1.0):
        super().__init__()
        self.use_focal = use_focal
        self.use_mse = use_mse
        self.adaptive_weight = adaptive_weight

        # 始终存储固定权重参数
        self.fixed_lambda_recon = lambda_recon
        self.fixed_lambda_class = lambda_class

        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        # 可学习的对数方差参数（初始化为0，对应 σ=1）
        if adaptive_weight:
            self.log_var_recon = nn.Parameter(torch.tensor([0.0]))  # σ=1
            self.log_var_class = nn.Parameter(torch.tensor([-1.0]))  # σ≈0.37，提高分类权重

        else:
            self.lambda_recon = lambda_recon
            self.lambda_class = lambda_class

    def forward(self, recon_ir, recon_raman, x_ir, x_raman, pred, labels):
        # 重构损失
        if self.use_mse:
            loss_recon_ir = self.mse(recon_ir, x_ir)
            loss_recon_raman = self.mse(recon_raman, x_raman)
        else:
            loss_recon_ir = F.l1_loss(recon_ir, x_ir)
            loss_recon_raman = F.l1_loss(recon_raman, x_raman)
        loss_recon = (loss_recon_ir + loss_recon_raman) / 2

        # 分类损失
        if self.use_focal:
            loss_class = self.focal_loss(pred, labels)
        else:
            loss_class = self.ce_loss(pred, labels)

        # 根据 adaptive_weight 选择计算方式
        if self.adaptive_weight:
            weight_recon = torch.exp(-self.log_var_recon)
            weight_class = torch.exp(-self.log_var_class)
            total_loss = (
                weight_recon * loss_recon + self.log_var_recon +
                weight_class * loss_class + self.log_var_class
            )
        else:
            total_loss = self.fixed_lambda_recon * loss_recon + self.fixed_lambda_class * loss_class

        return total_loss, loss_recon, loss_class