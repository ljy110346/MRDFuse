import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from models.encoder import EncoderWithBranch
from models.decoder import Decoder
from models.classifier import Classifier
from models.fusion import CrossModalAttentionFusion
from models.loss import JointLoss
import torch.nn.functional as F
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# 读取数据
train_ir = torch.tensor(pd.read_excel('data/banmo/train/train_ir.xlsx', header=None).values, dtype=torch.float32)
train_raman = torch.tensor(pd.read_excel('data/banmo/train/train_raman.xlsx', header=None).values, dtype=torch.float32)
train_labels = torch.tensor(pd.read_excel('data/banmo/train/train_labels.xlsx', header=None).values.flatten(), dtype=torch.long)

val_ir = torch.tensor(pd.read_excel('data/banmo/val/val_ir.xlsx', header=None).values, dtype=torch.float32)
val_raman = torch.tensor(pd.read_excel('data/banmo/val/val_raman.xlsx', header=None).values, dtype=torch.float32)
val_labels = torch.tensor(pd.read_excel('data/banmo/val/val_labels.xlsx', header=None).values.flatten(), dtype=torch.long)

# 封装数据集
batch_size = 32
train_dataset = TensorDataset(train_ir, train_raman, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_ir, val_raman, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义超参数
num_epochs = 120
learning_rate = 0.0001
z_dim = 128
attention_dim = 256
adaptive_weight = True
weight_decay = 1e-4  # L2 正则化强度

# **确保在初始化 Decoder 时传入 output_dim**
input_dim_ir = train_ir.shape[1]
input_dim_raman = train_raman.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
encoder_ir = EncoderWithBranch(input_dim_ir=input_dim_ir, input_dim_raman=input_dim_raman,z_dim=z_dim).to(device)
encoder_raman = EncoderWithBranch(input_dim_ir=input_dim_ir, input_dim_raman=input_dim_raman,z_dim=z_dim).to(device)
decoder_ir = Decoder(z_dim=z_dim, output_dim=input_dim_ir).to(device)
decoder_raman = Decoder(z_dim=z_dim, output_dim=input_dim_raman).to(device)
classifier = Classifier(input_dim=z_dim, num_classes=2).to(device)
fusion = CrossModalAttentionFusion(z_dim=z_dim,attention_dim=attention_dim).to(device)
criterion = JointLoss(lambda_recon=0.5, lambda_class=1.0, use_focal=True, use_mse=True,adaptive_weight=adaptive_weight).to(device)

# 定义优化器（增加 L2 正则化）
optimizer = optim.Adam(
    list(encoder_ir.parameters()) + list(encoder_raman.parameters()) +
    list(decoder_ir.parameters()) + list(decoder_raman.parameters()) +
    list(classifier.parameters()) + list(fusion.parameters()),
    lr=learning_rate,
    weight_decay = weight_decay  # L2 正则化
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

def train():
    # 存储训练过程中的指标
    train_losses, val_losses = [], []
    train_recon_losses, train_class_losses = [], []
    val_recon_losses, val_class_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    train_recalls, val_recalls = [], []
    all_train_z, all_train_labels = [], []
    all_val_z, all_val_labels = [], []


    for epoch in range(num_epochs):
        # ========== Adaptive Weight Warm-Up ==========
        if adaptive_weight:
            if epoch < 10:
                criterion.adaptive_weight = False  # 暂时关闭自适应
            else:
                criterion.adaptive_weight = True   # 启用自适应
        encoder_ir.train()
        encoder_raman.train()
        decoder_ir.train()
        decoder_raman.train()
        classifier.train()
        fusion.train()

        total_loss, total_correct, total_samples = 0, 0, 0
        total_recon_loss, total_class_loss = 0, 0
        all_preds, all_labels = [], []

        for x_ir, x_raman, labels in train_loader:
            x_ir, x_raman, labels = x_ir.to(device), x_raman.to(device), labels.to(device)
            optimizer.zero_grad()

            # 提取潜在特征
            z_ir_all = encoder_ir(x_ir, x_raman)
            z_raman_all = encoder_raman(x_ir, x_raman)

            # 各取子空间
            z_ir_raw = z_ir_all['z_raw']
            z_raman_raw = z_raman_all['z_raw']
            z_ir_cross = z_ir_all['z_cross']
            z_raman_cross = z_raman_all['z_cross']

            # 重构
            recon_x_ir = decoder_ir(z_ir_raw)
            recon_x_raman = decoder_raman(z_raman_raw)

            # 跨模态特征融合
            fused_z = fusion(z_ir_cross, z_raman_cross)

            # 分类
            preds = classifier(fused_z.float())

            # 计算损失

            loss, l_recon, l_class = criterion(recon_x_ir, recon_x_raman, x_ir, x_raman, preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += l_recon.item()
            total_class_loss += l_class.item()

            total_correct += (preds.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 存储 z 用于可视化
            all_train_z.append(fused_z.cpu().detach().numpy())
            all_train_labels.append(labels.cpu().numpy())

        train_f1_scores.append(f1_score(all_labels, all_preds, average='macro'))
        train_recalls.append(recall_score(all_labels, all_preds, average='macro'))
        train_accuracies.append(total_correct / total_samples)
        train_losses.append(total_loss / len(train_loader))
        train_recon_losses.append(total_recon_loss / len(train_loader))
        train_class_losses.append(total_class_loss / len(train_loader))
        # ========== Clip log_var 防止发散 ==========
        if adaptive_weight and criterion.adaptive_weight:
            criterion.log_var_recon.data.clamp_(-3.0, 3.0)
            criterion.log_var_class.data.clamp_(-3.0, 3.0)

        # **验证阶段**
        encoder_ir.eval()
        encoder_raman.eval()
        decoder_ir.eval()
        decoder_raman.eval()
        classifier.eval()
        fusion.eval()

        val_loss, val_correct, val_samples = 0, 0, 0
        val_recon_loss, val_class_loss = 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for x_ir, x_raman, labels in val_loader:
                x_ir, x_raman, labels = x_ir.to(device), x_raman.to(device), labels.to(device)
                # 提取潜在特征
                z_ir_all = encoder_ir(x_ir, x_raman)
                z_raman_all = encoder_raman(x_ir, x_raman)
                # 各取子空间
                z_ir_raw = z_ir_all['z_raw']
                z_raman_raw = z_raman_all['z_raw']
                z_ir_cross = z_ir_all['z_cross']
                z_raman_cross = z_raman_all['z_cross']
                # 重构
                recon_x_ir = decoder_ir(z_ir_raw)
                recon_x_raman = decoder_raman(z_raman_raw)
                fused_z = fusion(z_ir_cross, z_raman_cross)
                preds = classifier(fused_z.float())

                loss, l_recon, l_class = criterion(recon_x_ir, recon_x_raman, x_ir, x_raman, preds, labels)

                val_loss += loss.item()
                val_recon_loss += l_recon.item()
                val_class_loss += l_class.item()
                val_correct += (preds.argmax(dim=1) == labels).sum().item()
                val_samples += labels.size(0)
                all_val_preds.extend(preds.argmax(dim=1).cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                # 存储 z 用于可视化
                all_val_z.append(fused_z.cpu().detach().numpy())

        val_f1_scores.append(f1_score(all_val_labels, all_val_preds, average='macro'))
        val_recalls.append(recall_score(all_val_labels, all_val_preds, average='macro'))
        val_accuracies.append(val_correct / val_samples)
        val_losses.append(val_loss / len(val_loader))
        val_recon_losses.append(val_recon_loss / len(val_loader))
        val_class_losses.append(val_class_loss / len(val_loader))

        scheduler.step()  # 添加此行
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, '
              f'L_recon: {train_recon_losses[-1]:.4f}, L_class: {train_class_losses[-1]:.4f}, '
              f'Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, '
              f'Val Acc: {val_accuracies[-1]:.4f}, Val F1: {val_f1_scores[-1]:.4f}, Val Recall: {val_recalls[-1]:.4f}')

    # 保存最终模型
    final_model_state = {
        'encoder_ir': encoder_ir.state_dict(),
        'encoder_raman': encoder_raman.state_dict(),
        'decoder_ir': decoder_ir.state_dict(),
        'decoder_raman': decoder_raman.state_dict(),
        'classifier': classifier.state_dict(),
        'fusion': fusion.state_dict(),
    }
    torch.save(final_model_state, 'weights/model_banmo.pth')
    print('Final model saved to weights/model_banmo.pth')

    # **绘制训练曲线**
    metrics = {"Train Loss": train_losses, "Val Loss": val_losses,
               "Train Acc": train_accuracies, "Val Acc": val_accuracies,
               "Train F1": train_f1_scores, "Val F1": val_f1_scores,
               "Train Recall": train_recalls, "Val Recall": val_recalls}

    plt.figure(figsize=(12, 8))
    for key, values in metrics.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.show()

    # **PCA & t-SNE 可视化**
    def visualize(z_list, labels_list, title):
        """
        可视化特征（PCA + t-SNE）
        :param z_list: list of feature tensors
        :param labels_list: list of label tensors
        :param title: 标题
        """
        z = np.vstack(z_list)  # 变成 (N, feature_dim)
        labels = np.hstack(labels_list)  # 变成 (N,)

        print(f"{title} - Labels Unique Values: {np.unique(labels)}")
        print(f"{title} - Labels Count: {np.bincount(labels)}")

        # **PCA 降维**
        pca = PCA(n_components=2).fit_transform(z)

        # **t-SNE 降维**
        tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)

        # **绘图**
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(pca[:, 0], pca[:, 1], c=labels, cmap='coolwarm', alpha=0.7, edgecolors='k')
        plt.title(f'{title} - PCA')

        plt.subplot(1, 2, 2)
        plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='coolwarm', alpha=0.7, edgecolors='k')
        plt.title(f'{title} - t-SNE')
        plt.show()

    # **获取特征**
    z_ir_train = [
        encoder_ir(x_ir.unsqueeze(0).to(device), x_raman.unsqueeze(0).to(device))['z_cross']
        .cpu().detach().numpy()
        for x_ir, x_raman in zip(train_ir, train_raman)
    ]
    z_raman_train = [
        encoder_raman(x_ir.unsqueeze(0).to(device), x_raman.unsqueeze(0).to(device))['z_cross']
        .cpu().detach().numpy()
        for x_ir, x_raman in zip(train_ir, train_raman)
    ]
    z_fused_train = [
        fusion(
            encoder_ir(x_ir.unsqueeze(0).to(device), x_raman.unsqueeze(0).to(device))['z_cross'],
            encoder_raman(x_ir.unsqueeze(0).to(device), x_raman.unsqueeze(0).to(device))['z_cross']
        ).cpu().detach().numpy()
        for x_ir, x_raman in zip(train_ir, train_raman)
    ]

    # **转换成 NumPy 数组**
    train_labels_np = [y.cpu().numpy() for y in train_labels]

    # **可视化**
    visualize(z_ir_train, train_labels_np, 'Train IR Features')
    visualize(z_raman_train, train_labels_np, 'Train Raman Features')
    visualize(z_fused_train, train_labels_np, 'Train Fusion Features')

if __name__ == "__main__":
    train()
