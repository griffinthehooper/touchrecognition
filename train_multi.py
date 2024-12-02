import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from lstm_multi import SkeletonLSTM
from dataloader3 import SkeletonDataset

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Click', 'Index Click', 'Thumb Click', 'Both Click'],
                yticklabels=['No Click', 'Index Click', 'Thumb Click', 'Both Click'],
                annot_kws={'size': 16})
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def validate(net, validate_loader, loss_function, device):
    net.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(validate_loader, desc="Validating"):
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)  # 获取最大概率的类别
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(validate_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return val_loss, accuracy, f1, precision, recall, all_preds, all_labels

def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 路径配置
    train_file_path = 'E:/Coding_file/Python/CNN/data/train_data3.csv'
    validate_file_path = 'E:/Coding_file/Python/CNN/data/val_data3.csv'
    results_path = './results'
    weights_path = os.path.join(results_path, 'weights')
    plots_path = os.path.join(results_path, 'plots')
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    # 数据加载
    train_dataset = SkeletonDataset(train_file_path, normalize=True)
    validate_dataset = SkeletonDataset(validate_file_path, normalize=True)
    
    # 计算类别数量
    train_labels = train_dataset.labels
    class_counts = torch.bincount(train_labels)
    print("\nClass distribution:")
    for i in range(len(class_counts)):
        print(f"Class {i}: {class_counts[i]} samples")

    # 计算类别权重
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    print("\nClass weights:")
    for i in range(len(class_weights)):
        print(f"Class {i}: {class_weights[i]:.4f}")
    
    # 创建加权采样器处理类别不平衡
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))


    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(validate_dataset)}")
    print(f"Class distribution in training set: {class_counts.numpy()}")

    # 模型初始化
    model = SkeletonLSTM().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # 使用计算好的类别权重
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    # 在训练循环前添加数据验证
    print("\nValidating first batch:")
    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        print("Batch data shape:", batch_data.shape)
        print("Batch labels shape:", batch_labels.shape)
        print("Batch labels min:", batch_labels.min().item())
        print("Batch labels max:", batch_labels.max().item())
        print("Unique labels in batch:", torch.unique(batch_labels).cpu().numpy())
        
        # 测试一次前向传播
        with torch.no_grad():
            outputs = model(batch_data)
            print("Model output shape:", outputs.shape)
            print("Model output example:", outputs[0])
        break

    # 训练参数
    epochs = 100
    best_f1 = 0.0
    patience = 10
    no_improve_epochs = 0

    # 记录训练过程
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_metrics = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        for data, labels in train_bar:
            try:
                # 移动数据到设备
                data = data.to(device)
                labels = labels.to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(data)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print("\nError during training:")
                print("Current batch labels:", labels)
                print("Model outputs shape:", outputs.shape)
                print("Labels shape:", labels.shape)
                print("Unique labels in batch:", torch.unique(labels).cpu().numpy())
                raise e

        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证
        val_loss, val_accuracy, val_f1, val_precision, val_recall, val_preds, val_labels = \
            validate(model, validate_loader, criterion, device)
        
        val_losses.append(val_loss)
        val_metrics.append({
            'accuracy': val_accuracy,
            'f1': val_f1,
            'precision': val_precision,
            'recall': val_recall
        })

        # 打印当前epoch的训练结果
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        scheduler.step()

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(weights_path, "best_model.pth"))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # 早停
        if no_improve_epochs >= patience:
            print(f"\nNo improvement for {patience} epochs. Stopping training.")
            break

    print('Training completed')

    # 绘制训练过程
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 指标曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, [m['accuracy'] for m in val_metrics], label='Val Accuracy')
    plt.plot(epochs_range, [m['f1'] for m in val_metrics], label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'training_history.png'))
    plt.show()

    # 绘制最终的混淆矩阵
    plot_confusion_matrix(val_labels, val_preds, 
                         save_path=os.path.join(plots_path, 'confusion_matrix.png'))

    # 打印最终的评估指标
    print("\nFinal Evaluation Metrics:")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Validation Precision: {val_precision:.4f}")
    print(f"Final Validation Recall: {val_recall:.4f}")

if __name__ == '__main__':
    main()