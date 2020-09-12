import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        #print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # filter = 2, default stride = filter
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #view [16,5,5] to [1,400]
        x = x.view(-1, self.num_flat_features(x))#[1,400]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x)
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
