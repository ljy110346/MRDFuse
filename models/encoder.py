import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- SE 模块 -----------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.se(x).unsqueeze(-1)
        return x * weights

# ----------------- 残差块 -----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, use_se=False):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.gelu(x + residual)

# ----------------- 模态感知门控（池化后） -----------------
class GatedFeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_ir = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.gate_raman = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, ir_feat, raman_feat):
        ir_input = torch.cat([ir_feat, raman_feat], dim=-1)
        raman_input = torch.cat([raman_feat, ir_feat], dim=-1)

        ir_gate = self.gate_ir(ir_input)
        raman_gate = self.gate_raman(raman_input)

        ir_updated = ir_feat + ir_gate * raman_feat
        raman_updated = raman_feat + raman_gate * ir_feat

        return ir_updated, raman_updated

# ----------------- 主体 Encoder -----------------
class EncoderWithBranch(nn.Module):
    def __init__(self, input_dim_ir, input_dim_raman, z_dim=128):
        super().__init__()
        self.z_dim = z_dim

        self.pre_ir = nn.Sequential(nn.Linear(input_dim_ir, 256), nn.GELU(), nn.LayerNorm(256))
        self.pre_raman = nn.Sequential(nn.Linear(input_dim_raman, 256), nn.GELU(), nn.LayerNorm(256))

        self.initial_conv_ir = nn.Sequential(nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU())
        self.initial_conv_raman = nn.Sequential(nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU())

        self.block1 = ResidualBlock(32, 64, dilation=1)
        self.block2 = ResidualBlock(64, 128, dilation=2)
        self.block3 = ResidualBlock(128, 128, use_se=True)
        self.downsample1 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)

        # 池化后的感知交叉模块
        self.cross_pool = GatedFeatureFusion(dim=128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, z_dim)
        )

        self.z_raw_branch = nn.Linear(z_dim, z_dim)
        self.z_cross_branch = nn.Linear(z_dim, z_dim)

    def forward(self, x_ir, x_raman):
        ir = self.pre_ir(x_ir).unsqueeze(1)  # [B, 256] -> [B, 1, 256]
        ir = self.initial_conv_ir(ir)

        raman = self.pre_raman(x_raman).unsqueeze(1)  # [B, 256] -> [B, 1, 256]
        raman = self.initial_conv_raman(raman)

        ir = self.block1(ir)
        raman = self.block1(raman)

        ir = self.downsample1(ir)
        raman = self.downsample1(raman)

        ir = self.block2(ir)
        raman = self.block2(raman)
        ir = self.downsample2(ir)
        raman = self.downsample2(raman)

        ir = self.block3(ir)
        raman = self.block3(raman)

        pooled_ir = self.pool(ir).squeeze(-1)      # [B, 128]
        pooled_raman = self.pool(raman).squeeze(-1)

        # 全连接空间的交互
        ir_updated, raman_updated = self.cross_pool(pooled_ir, pooled_raman)

        feat = (ir_updated + raman_updated) / 2     # 融合

        z_base = self.projector(feat)

        z_raw = self.z_raw_branch(z_base)
        z_cross = self.z_cross_branch(z_base)

        return {
            'z_base': z_base,
            'z_raw': z_raw,
            'z_cross': z_cross
        }