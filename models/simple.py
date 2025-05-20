import torch.nn as nn
import torch.nn.functional as F

class Lenet5(nn.Module):
    def __init__(self, cnn_features1=6, cnn_features2=16):
        super(Lenet5, self).__init__()
        self.cnn_features1 = cnn_features1 
        self.cnn_features2 = cnn_features2 

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=cnn_features1, kernel_size=5) 
        self.conv2 = nn.Conv2d(in_channels=cnn_features1, out_channels=cnn_features2, kernel_size=5)

        self.fc1 = nn.Linear(cnn_features2*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        
        x = self.relu2(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(-1, self.cnn_features2*5*5)

        x = self.fc1(x)
        
        x = self.relu3(x)

        x = self.fc2(x)
        
        x = self.relu4(x)

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x