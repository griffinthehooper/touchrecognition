import torch
import torch.nn as nn

class SkeletonLSTM(nn.Module):
    def __init__(self, input_channels=2, num_joints=21, hidden_size=128, num_layers=2, num_classes=4):
        super(SkeletonLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.num_joints = num_joints
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes  # 添加这行，明确保存类别数
        
        # 空间特征提取
        self.spatial_features = nn.Sequential(
            nn.Linear(input_channels * num_joints, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=128,  # 空间特征的输出大小
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # 分类层 - 修改最后一层输出维度为num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2是因为双向LSTM
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  # 确保输出维度为类别数量
        )

    def forward(self, x):
        # 输入x的形状: (batch_size, channels, joints, frames)
        batch_size = x.size(0)
        
        # 添加形状检查
        assert x.size(1) == self.input_channels, f"Expected {self.input_channels} channels, got {x.size(1)}"
        assert x.size(2) == self.num_joints, f"Expected {self.num_joints} joints, got {x.size(2)}"
        
        # 重新排列维度，准备进行空间特征提取
        # (batch_size, frames, channels * joints)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 5, -1)
        
        # 对每个时间步进行空间特征提取
        spatial_out = []
        for t in range(x.size(1)):
            spatial_features = self.spatial_features(x[:, t, :])
            spatial_out.append(spatial_features)
        
        # 堆叠所有时间步的空间特征
        x = torch.stack(spatial_out, dim=1)  # (batch_size, frames, 128)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 获取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 分类
        out = self.classifier(lstm_out)
        
        # 添加输出形状检查
        assert out.size(1) == self.num_classes, f"Expected output size {self.num_classes}, got {out.size(1)}"
        
        return out

    def init_hidden(self, batch_size, device):
        """初始化LSTM隐藏状态"""
        h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)