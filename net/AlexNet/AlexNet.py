import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
             #3x227x227
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        #自适应均值池化，给定输出的情况下，会自动将输入数据池化到固定尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def alexnet(self, pretrained=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = torch.load('alexnet-owt-4df8aa71.pth')
        model.load_state_dict(state_dict)
    return model




'''
alexNext = AlexNet(1000)
inputsize = 64
input = torch.randn(1, 3, inputsize, inputsize)
print('#input:\n',input.shape)
features = alexNext.features

print('#net:\n',features)
print('#output:')
for i in range(13):
    input = features[i](input)
    print(i,input.shape)
'''
#input = alexNext.avgpool(input)
#print(input.shape)

