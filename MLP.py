import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
from datetime import datetime

# 设置随机种子保证可重复性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 自定义数据集类（包含标准化）
class TabularDataset(Dataset):
    def __init__(self, features, targets, scaler=None):
        """
        参数:
        features: DataFrame 特征数据
        targets: Series 目标变量
        scaler: 可选，预训练的标准化器
        """
        self.features = features.values.astype(np.float32)
        self.targets = targets.values.astype(np.long)

        # 特征标准化
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.long)

    def get_scaler(self):
        return self.scaler


# 改进的神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size

        # 动态构建隐藏层
        for i, h_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h_size

        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 训练验证函数
def train_val_model(model, train_loader, val_loader, criterion, optimizer,
                    scheduler=None, epochs=100, patience=10, log_dir=None):
    """
    训练验证模型，包含早停和模型保存
    返回:
    best_model: 最佳模型
    history: 训练历史记录
    """
    best_val_loss = float('inf')
    best_model = None
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # TensorBoard日志
    writer = SummaryWriter(log_dir) if log_dir else None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        train_loss /= len(train_loader.dataset)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 调整学习率
        if scheduler:
            scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            no_improve = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_model.pth')
        else:
            no_improve += 1

        # 打印进度
        print(f'Epoch {epoch + 1}/{epochs} - '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}%')

        # TensorBoard记录
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model, history


def evaluate_model(model, data_loader, criterion=None):
    """
    评估模型性能
    返回:
    loss: 平均损失（如果提供criterion）
    accuracy: 准确率
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader.dataset) if criterion else None
    return avg_loss, accuracy


def get_metrics(model, data_loader, class_names):
    """
    获取详细评估指标
    返回分类报告、混淆矩阵和AUC
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 计算指标
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr') if len(class_names) > 2 \
        else roc_auc_score(all_labels, all_probs[:, 1])

    return report, cm, auc


if __name__ == "__main__":
    # 数据准备
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("inter.csv")

    # 特征和目标列配置
    FEATURE_COLS = train_df.columns[2:].tolist()  # 假设前两列是ID和目标
    TARGET_COL = train_df.columns[1]

    # 标签编码
    label_map = {1: 0, 2: 1, 4: 2}  # 根据实际类别修改
    train_df[TARGET_COL] = train_df[TARGET_COL].map(label_map)
    val_df[TARGET_COL] = val_df[TARGET_COL].map(label_map)

    # 创建数据集
    train_dataset = TabularDataset(
        train_df[FEATURE_COLS],
        train_df[TARGET_COL]
    )
    val_dataset = TabularDataset(
        val_df[FEATURE_COLS],
        val_df[TARGET_COL],
        scaler=train_dataset.get_scaler()
    )

    # 保存标准化器
    joblib.dump(train_dataset.get_scaler(), 'scaler.pkl')

    # 创建数据加载器
    BATCH_SIZE = 64
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4
    )

    # 模型配置
    INPUT_SIZE = len(FEATURE_COLS)
    HIDDEN_SIZES = [128, 64, 32]
    NUM_CLASSES = len(label_map)
    DROPOUT = 0.3

    model = NeuralNet(
        INPUT_SIZE, HIDDEN_SIZES,
        NUM_CLASSES, DROPOUT
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )

    # 训练参数
    EPOCHS = 200
    PATIENCE = 15
    LOG_DIR = f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 训练模型
    print("Starting training...")
    best_model, history = train_val_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        epochs=EPOCHS, patience=PATIENCE,
        log_dir=LOG_DIR
    )

    # 最终评估
    print("\nEvaluating best model...")
    _, val_acc = evaluate_model(best_model, val_loader)
    report, cm, auc = get_metrics(best_model, val_loader, list(label_map.keys()))

    print(f"\nValidation Accuracy: {val_acc:.2f}%")
    print(f"Validation AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    # 保存完整模型
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'scaler': train_dataset.get_scaler(),
        'feature_cols': FEATURE_COLS,
        'label_map': label_map,
        'model_params': {
            'input_size': INPUT_SIZE,
            'hidden_sizes': HIDDEN_SIZES,
            'num_classes': NUM_CLASSES,
            'dropout': DROPOUT
        }
    }, "final_model.pth")