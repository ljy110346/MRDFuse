import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------- 简化版 SE 注意力 -----------
class SimpleSE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

# ----------- Decoder 主体结构 -----------
class Decoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim

        # z → 高维特征 → Reshape 为 conv 输入
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256 * 4),
            nn.GELU()
        )

        # 反卷积 + 注意力模块（匹配 encoder 维度结构）
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.att1 = SimpleSE(128)

        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.att2 = SimpleSE(64)

        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.att3 = SimpleSE(32)

        self.deconv4 = nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

        # 动态输出匹配 final_fc（确保可对接任意维度模态）
        self.final_fc = nn.Linear(128, output_dim)

    def forward(self, z, output_dim=None, skip_features=None):
        """
        z: 来自 encoder.z_raw
        output_dim: 动态决定最终光谱维度
        skip_features: 为未来可能的 skip-connection 留接口（当前未使用）
        """
        x = self.fc(z)                          # [B, 1024]
        x = x.view(x.size(0), 256, 4)           # [B, 256, 4]

        x = self.deconv1(x)                     # [B, 128, 8]
        x = self.att1(x)

        x = self.deconv2(x)                     # [B, 64, 16]
        x = self.att2(x)

        x = self.deconv3(x)                     # [B, 32, 32]
        x = self.att3(x)

        x = self.deconv4(x)                     # [B, 1, 64]（假设初始为 4，经过四次×2）

        x = x.view(x.size(0), -1)               # [B, L]

        final_dim = output_dim if output_dim is not None else self.output_dim
        if x.shape[1] != final_dim:
            self.final_fc = nn.Linear(x.shape[1], final_dim).to(x.device)

        return self.final_fc(x)
