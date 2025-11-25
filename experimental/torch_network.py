import torch.nn as nn

class Conv3DModel(nn.Module):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64*4*4*4, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(-1, 64*4*4*4)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x



