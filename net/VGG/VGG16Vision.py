import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vgg import vgg16

from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataLoader import getAnimalDataloader,getSmokeDataloader,getFireDataloader
from collections import OrderedDict
import torch
import math
import cv2
import layervision

#-------------加载网络结构----------

path = '../models/'+'VGG_fire173.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path):
    net = vgg16(pretrained=False,progress=False)
    classifier = nn.Sequential(OrderedDict([
                                    ('fc1',nn.Linear(25088,4096)),
                                     ('relu1',nn.ReLU()),
                                     ('fc2',nn.Linear(4096,1000)),
                                     ('relu2',nn.ReLU()),
                                     ('fc3',nn.Linear(1000,10)),
                                     ('output',nn.LogSoftmax(dim=1))
        ]))
        # 替换
    net.classifier = classifier
    net.to(device)
    net.load_state_dict(torch.load(path))
    return net

model = load_model(path)

print("model:\n",model)


#定义数据加载器

_,test_loader = getFireDataloader(227,1)

# 随机获取训练图片
dataiter = iter(test_loader)







 
images, labels = dataiter.next()

#features各层输出
if True:
    output_dir = "./output"
    images = images.to(device)
    mid_output = layervision.write_con2d_pool_layers_output(images, model.features, output_dir)
    mid_output = model.avgpool(mid_output.to(device))
    mid_output = torch.flatten(mid_output.to(device),1)
    layervision.write_linear_layers_output(mid_output, model.classifier, output_dir)








