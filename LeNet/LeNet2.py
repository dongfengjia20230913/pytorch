import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self,imgChannel):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(imgChannel, 6, 5),#self.C1 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),# self.S2

            nn.Conv2d(6, 16, 5),#self.C3 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#self.S4
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),#self.fc1
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),#self.fc2
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),#self.fc3
        )

    def forward(self, x):
      
        x = self.features(x)
        x = x.view(-1, self.num_flat_features(x))#[1,400]
        x =self.classifier(x)
        return x

    def num_flat_features(self, x):
        # 除去批处理维度的其他所有维度
        #torch.Size([1, 16, 5, 5])--->torch.Size([16, 5, 5])
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features#400


#net = Net()
#print(net)

#input = torch.randn(1, 1, 32, 32)
#net.forward(input)
