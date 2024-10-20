import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns

class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SkeletonDataset(Dataset):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self.data = pd.read_csv(file_path, header=None)
        print(f"Data shape: {self.data.shape}")
        print(f"Number of features: {self.data.shape[1] - 1}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = torch.tensor(row[0], dtype=torch.float)
        features = torch.tensor(row[1:].values, dtype=torch.float)
        return features, label

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def validate(net, validate_loader, loss_function, device):
    net.eval()
    acc_num = 0.0
    epoch_val_losses = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in validate_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            loss = loss_function(outputs, labels.unsqueeze(1))
            epoch_val_losses.append(loss.item())
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc_num += torch.sum(preds == labels.unsqueeze(1).float())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss_avg = sum(epoch_val_losses) / len(validate_loader)
    val_acc = acc_num.item() / len(validate_loader.dataset)
    
    return val_loss_avg, val_acc, all_preds, all_labels

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_file_path = os.getenv('TRAIN_FILE_PATH', 'E:/Coding_file/Python/CNN/data/window_data3.csv')
    validate_file_path = os.getenv('VALIDATE_FILE_PATH', 'E:/Coding_file/Python/CNN/data/window_data2.csv')

    train_dataset = SkeletonDataset(train_file_path)
    validate_dataset = SkeletonDataset(validate_file_path)
    
    sample_data, sample_label = train_dataset[0]
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample label shape: {sample_label.shape}")

    input_size = sample_data.shape[0]
    net = SimpleCNN(input_size).to(device)
    
    from torchsummary import summary
    summary(net, (input_size,))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)

    print(f"using {len(train_dataset)} samples for training, {len(validate_dataset)} samples for validation.")

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    epochs = 100
    best_acc = 0.0
    save_path = 'E:/Coding_file/Python/CNN/results/weights/SimpleCNN'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_losses, train_accuracies, val_accuracies, val_losses = [], [], [], []
    patience, no_improve_epochs = 10, 0
    all_preds, all_labels = [], []  # 用于存储所有的预测和标签

    for epoch in range(epochs):
        net.train()
        acc_num = torch.zeros(1).to(device)
        train_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)

        for data, labels in train_bar:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            loss = loss_function(outputs, labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc_num += torch.sum(preds == labels.unsqueeze(1).float())
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_acc = acc_num.item() / len(train_dataset)
        train_loss_avg = train_loss / len(train_loader)
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        print(f'[epoch {epoch + 1}] train_loss: {train_loss_avg:.3f}  train_acc: {train_acc:.3f}')

        val_loss_avg, val_acc, epoch_preds, epoch_labels = validate(net, validate_loader, loss_function, device)
        
        # 收集每个epoch的预测和标签
        all_preds.extend(epoch_preds)
        all_labels.extend(epoch_labels)

        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)
        print(f'[epoch {epoch + 1}] val_loss: {val_loss_avg:.3f}  val_accuracy: {val_acc:.3f}')
        
        scheduler.step(val_loss_avg)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), os.path.join(save_path, "SimpleCNN.pth"))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print('Finished Training')

    # 训练结束后绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, classes=['0', '1'])
    
    # 计算并打印最终的每个类别准确率
    cm = confusion_matrix(all_labels, all_preds)
    class_0_accuracy = cm[0,0] / (cm[0,0] + cm[0,1])
    class_1_accuracy = cm[1,1] / (cm[1,0] + cm[1,1])
    print(f"Final Class 0 Accuracy: {class_0_accuracy:.4f}")
    print(f"Final Class 1 Accuracy: {class_1_accuracy:.4f}")

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plot_save_path = os.getenv('PLOT_SAVE_PATH', './results/plots')
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    plt.savefig(os.path.join(plot_save_path, 'training_metrics.png'))
    plt.show()

if __name__ == '__main__':
    main()