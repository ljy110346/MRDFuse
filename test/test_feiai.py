import torch
import pandas as pd
import torch.nn.functional as F
from models.encoder import EncoderWithBranch
from models.classifier import Classifier
from models.fusion import CrossModalAttentionFusion
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np
import matplotlib
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
# 设置全局字体
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12  # 统一控制字号

# **1. 读取测试数据**
test_ir_path = "data/feiai/test/test_ir.xlsx"
test_raman_path = "data/feiai/test/test_raman.xlsx"
test_labels_path = "data/feiai/test/test_labels.xlsx"

df_ir = pd.read_excel(test_ir_path, header=None)
df_raman = pd.read_excel(test_raman_path, header=None)
df_labels = pd.read_excel(test_labels_path, header=None)

# **2. 转换为 PyTorch 张量**
x_ir = torch.tensor(df_ir.values, dtype=torch.float32)
x_raman = torch.tensor(df_raman.values, dtype=torch.float32)
y_true = torch.tensor(df_labels.values, dtype=torch.long).squeeze()  # 确保为 1D 张量

num_samples = x_ir.shape[0]  # 获取测试集样本数
input_dim_ir = x_ir.shape[1]
input_dim_raman = x_raman.shape[1]

# **3. 设定与训练时一致的参数**
z_dim = 128  # 训练时的潜在空间维度
attention_dim = 256  # 交叉注意力层的维度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **4. 加载模型**
encoder_ir = EncoderWithBranch(input_dim_ir=input_dim_ir, input_dim_raman=input_dim_raman, z_dim=z_dim).to(device)
encoder_raman = EncoderWithBranch(input_dim_ir=input_dim_ir, input_dim_raman=input_dim_raman, z_dim=z_dim).to(device)

classifier = Classifier(input_dim=z_dim, num_classes=2).to(device)
fusion = CrossModalAttentionFusion(z_dim=z_dim, attention_dim=attention_dim).to(device)

# **5. 加载已训练的模型参数**
checkpoint = torch.load("weights/model_feiai.pth", map_location=device)
encoder_ir.load_state_dict(checkpoint['encoder_ir'])
encoder_raman.load_state_dict(checkpoint['encoder_raman'])

classifier.load_state_dict(checkpoint['classifier'])
fusion.load_state_dict(checkpoint['fusion'])

# **6. 进入评估模式**
encoder_ir.eval()
encoder_raman.eval()
fusion.eval()
classifier.eval()

# 7. 测试运行中添加特征记录
z_fusion_list = []
y_pred = []

with torch.no_grad():
    for i in range(num_samples):
        # 确保两个模态都传入，与训练保持一致
        sample_ir = x_ir[i].unsqueeze(0).to(device)
        sample_raman = x_raman[i].unsqueeze(0).to(device)

        # 使用两个输入，且提取字典中特定分支的特征
        z_ir_dict = encoder_ir(sample_ir, sample_raman)
        z_raman_dict = encoder_raman(sample_ir, sample_raman)
        z_ir = z_ir_dict['z_cross']
        z_raman = z_raman_dict['z_cross']

        # 融合两边的特征
        z_fusion = fusion(z_ir, z_raman)
        output = classifier(z_fusion)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).item()
        y_pred.append(pred)
        z_fusion_list.append(z_fusion.squeeze(0).cpu().numpy())

# **8. 计算评估指标**
y_true_np = y_true.numpy()
y_pred_np = np.array(y_pred)

# 获取预测概率用于 AUC 与 ROC
y_prob = []
with torch.no_grad():
    for i in range(num_samples):
        sample_ir = x_ir[i].unsqueeze(0).to(device)
        sample_raman = x_raman[i].unsqueeze(0).to(device)
        z_ir = encoder_ir(sample_ir, sample_raman)['z_cross']
        z_raman = encoder_raman(sample_ir, sample_raman)['z_cross']
        z_fusion = fusion(z_ir, z_raman)
        output = classifier(z_fusion)
        prob = F.softmax(output, dim=1)[:, 1].item()  # 获取正类概率
        y_prob.append(prob)

y_prob_np = np.array(y_prob)

# 混淆矩阵（TN, FP, FN, TP）
cm = confusion_matrix(y_true_np, y_pred_np)
tn, fp, fn, tp = cm.ravel()

# 指标计算
accuracy = accuracy_score(y_true_np, y_pred_np)
precision = precision_score(y_true_np, y_pred_np, pos_label=1)
sensitivity = recall_score(y_true_np, y_pred_np, pos_label=1)
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
f1 = f1_score(y_true_np, y_pred_np, pos_label=1)
auc = roc_auc_score(y_true_np, y_prob_np)

# **9. 打印评估结果**
print("\n===== 测试集评估结果（排序输出） =====")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"AUC        : {auc:.4f}")

# 10. 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_true_np, y_prob_np)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
