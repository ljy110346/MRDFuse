import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, z_dim, attention_dim):
        super().__init__()
        # 增加门控机制
        self.gate = nn.Sequential(
            nn.Linear(2 * z_dim, z_dim),
            nn.Sigmoid()
        )

        self.z_dim = z_dim

        # IR -> Raman 注意力参数
        self.query_fc_ir = nn.Linear(z_dim, attention_dim)
        self.key_fc_raman = nn.Linear(z_dim, attention_dim)
        self.value_fc_raman = nn.Linear(z_dim, z_dim)

        # Raman -> IR 注意力参数
        self.query_fc_raman = nn.Linear(z_dim, attention_dim)
        self.key_fc_ir = nn.Linear(z_dim, attention_dim)
        self.value_fc_ir = nn.Linear(z_dim, z_dim)

        # 动态门控融合
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * z_dim, z_dim),
            nn.Sigmoid()
        )

        # 正则化与稳定性
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.layer_norm = nn.LayerNorm(z_dim)  # 修改为单一输出维度
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, z_ir, z_raman):
        # IR -> Raman 注意力
        query_ir = self.query_fc_ir(z_ir).unsqueeze(1)        # [B,1,A]
        key_raman = self.key_fc_raman(z_raman).unsqueeze(2)    # [B,A,1]
        value_raman = self.value_fc_raman(z_raman)             # [B,Z]
        attn_scores_ir = torch.bmm(query_ir, key_raman) / (self.z_dim**0.5)
        attn_weights_ir = F.softmax(attn_scores_ir, dim=-1)    # [B,1,1]
        attn_output_ir = attn_weights_ir.squeeze(1) * value_raman  # [B,Z]

        # Raman -> IR 注意力
        query_raman = self.query_fc_raman(z_raman).unsqueeze(1)
        key_ir = self.key_fc_ir(z_ir).unsqueeze(2)
        value_ir = self.value_fc_ir(z_ir)
        attn_scores_raman = torch.bmm(query_raman, key_ir) / (self.z_dim**0.5)
        attn_weights_raman = F.softmax(attn_scores_raman, dim=-1)
        attn_output_raman = attn_weights_raman.squeeze(1) * value_ir

        # 残差连接与门控融合
        fused_ir = self.gamma * attn_output_ir + (1 - self.gamma) * z_ir
        fused_raman = self.gamma * attn_output_raman + (1 - self.gamma) * z_raman
        combined = torch.cat([fused_ir, fused_raman], dim=-1)
        gate = self.gate_fc(combined)
        fused_data = gate * fused_ir + (1 - gate) * fused_raman

        # 归一化与正则化
        fused_data = self.layer_norm(fused_data)
        return self.dropout(fused_data)