import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from network import VGG16
from dataloader import SkeletonDataset, get_sampler

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
            labels = labels.unsqueeze(1)
            loss = loss_function(outputs, labels)
            epoch_val_losses.append(loss.item())
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc_num += torch.sum(preds == labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss_avg = sum(epoch_val_losses) / len(validate_loader)
    val_acc = acc_num.item() / len(validate_loader.dataset)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    return val_loss_avg, val_acc, all_preds, all_labels, f1, precision, recall

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_file_path = 'E:/Coding_file/Python/CNN/data/output2.csv'
    validate_file_path = 'E:/Coding_file/Python/CNN/data/output.csv'

    train_dataset = SkeletonDataset(train_file_path)
    validate_dataset = SkeletonDataset(validate_file_path)
    
    sample_data, sample_label = train_dataset[0]
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample label shape: {sample_label.shape}")

    net = VGG16().to(device)

    sampler, class_counts = get_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    validate_loader = DataLoader(validate_dataset, batch_size=16, shuffle=False)

    print(f"using {len(train_dataset)} samples for training, {len(validate_dataset)} samples for validation.")

    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-5)  
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    epochs = 100
    best_acc = 0.0
    save_path = './results/weights/VGGSkeleton'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_losses, train_accuracies, val_accuracies, val_losses = [], [], [], []
    patience, no_improve_epochs = 20, 0
    all_preds, all_labels = [], []

    for epoch in range(epochs):
        net.train()
        acc_num = torch.zeros(1).to(device)
        train_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)

        for data, labels in train_bar:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            labels = labels.unsqueeze(1)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc_num += torch.sum(preds == labels)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_acc = acc_num.item() / len(train_dataset)
        train_loss_avg = train_loss / len(train_loader)
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        print(f'[epoch {epoch + 1}] train_loss: {train_loss_avg:.3f}  train_acc: {train_acc:.3f}')
    
        val_loss_avg, val_acc, epoch_preds, epoch_labels, f1, precision, recall = validate(net, validate_loader, loss_function, device)

        all_preds.extend(epoch_preds)
        all_labels.extend(epoch_labels)

        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)
        print(f'[epoch {epoch + 1}] val_loss: {val_loss_avg:.3f}  val_accuracy: {val_acc:.3f}')
        print(f'F1: {f1:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}')
        
        scheduler.step(val_loss_avg)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), os.path.join(save_path, "VGGSkeleton.pth"))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print('Finished Training')

    plot_confusion_matrix(all_labels, all_preds, classes=['0', '1'])
    
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

    plot_save_path = './results/plots'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    plt.savefig(os.path.join(plot_save_path, 'training_metrics.png'))
    plt.show()

if __name__ == '__main__':
    main()