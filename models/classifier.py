import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class ResidualLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),  # 更适合小batch
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return x + self.block(x)

class Classifier(nn.Module):
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dims: list = [256, 128],
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 use_se: bool = True,
                 use_residual: bool = True):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 替换部分 BatchNorm
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))

            if use_residual:
                layers.append(ResidualLayer(hidden_dim))
            if use_se:
                layers.append(SEBlock(hidden_dim))

            prev_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z):
        x = self.feature_layers(z)
        logits = self.output_layer(x)
        return logits