import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import LeNet2
import numpy as np
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image

#-------------加载网络结构----------

path = '../models/'+'lenet_3.pth'

def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LeNet2.Net().to(device)
    net.load_state_dict(torch.load(path))
    return net

model = load_model(path)

print(model.features)

#定义数据加载器
resize = 32
transform = transforms.Compose([transforms.Resize(size=(resize, resize)),
                                    torchvision.transforms.ToTensor()
                                    ])
test_data = torchvision.datasets.MNIST(root="../datas",
                                            train=False,
                                            transform=transform,
                                            download=False)
test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                            batch_size = 1,
                                            shuffle = False)
#-------------显示处理前的图片----------
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 随机获取训练图片
dataiter = iter(test_loader)
images, labels = dataiter.next()

# 显示图片
#imshow(torchvision.utils.make_grid(images))

#-----------保存各层处理后的图片-------------------

def save_img(tensor, name):
    #替换深度和batch_size所在的纬度值
    tensor = tensor.permute((1, 0, 2, 3))#将[1, 6, 28, 28]转化成[1, 6, 28, 28]
    print('output permute:',tensor.shape)
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.cpu().data.numpy() * 255.).astype(np.uint8)#将0~1之间的像素值，转化成0~255
    Image.fromarray(im).save(name + '.jpg')
    
def save_img_linear(tensor, name):
    #替换深度和batch_size所在的纬度值
    tensor = tensor.permute((1, 0))
    print('output permute:',tensor.shape)
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.cpu().data.numpy() * 255.).astype(np.uint8)#将0~1之间的像素值，转化成0~255
    Image.fromarray(im).save(name + '.jpg')

#模型训练的时候，图片是加载到GPU进行训练的，此时的前馈过程输入格式要跟训练时保持一致
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#features各层输出
for i in range(6):
    print('------features%d------'%i)
    new_model = model.features[i]
    print('input:',images.shape)

    print('layer:',new_model)
    layer_out = new_model(images.to(device))
    print('output',layer_out.shape)
    save_img(layer_out, 'features'+str(i))
    images = layer_out #下一层网络的输入是上一层网络的输出

#将features最后一层输出的数据[1,16,5,5]转化成[1,400]
images = images.view(-1, model.num_flat_features(images))
#classifier各层输出
for i in range(5):
    print('------classifier%d------'%i)
    new_model = model.classifier[i]
    print('input:',images.shape)

    print('layer:',new_model)
    layer_out = new_model(images.to(device))
    print('output',layer_out.shape)
    save_img_linear(layer_out, 'classifier'+str(i))
    images = layer_out #下一层网络的输入是上一层网络的输出













 








