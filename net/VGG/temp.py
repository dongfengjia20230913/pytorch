import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vgg import vgg16
import numpy as np
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from dataLoader import getAnimalDataloader,getSmokeDataloader,getFireDataloader
from collections import OrderedDict
import torch
import math
import cv2

#-------------加载网络结构----------

path = '../models/'+'VGG_fire15.pth'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
#-------------显示处理前的图片----------
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 随机获取训练图片
dataiter = iter(test_loader)
images, labels = dataiter.next()

# 显示图片
#imshow(torchvision.utils.make_grid(images))

#-----------保存各层处理后的图片-------------------

def save_img_gray(tensor, name):
    #替换深度和batch_size所在的纬度值
    tensor = tensor.permute((1, 0, 2, 3))#将[1, 6, 28, 28]转化成[1, 6, 28, 28]
    print('output permute:',tensor.shape)
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.cpu().data.numpy() * 255.).astype(np.uint8)#将0~1之间的像素值，转化成0~255
    Image.fromarray(im).save(name + '.jpg')



def save_img_heatmap(tensor, name):
    #替换深度和batch_size所在的纬度值
    #tensor = tensor.permute((1, 0, 2, 3))#将[1, 6, 28, 28]转化成[6, 1, 28, 28]
    temp_img = tensor.cpu().detach().permute((1, 0, 2, 3))
    print('output permute----------:',temp_img.size())
    featureMapSize = temp_img.size()[0]
    row = 8
    line = math.ceil(featureMapSize/8)
    imagesList=[]
    for i in range(featureMapSize):
        img1 = temp_img[i][0]
        #print(i,img1.shape)
        plt.subplot(line,row,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img1)
        imagesList.append(img1)
    plt.savefig(name,dpi=300)

def save_img_linear(tensor, name):
    #替换深度和batch_size所在的纬度值
    tensor = tensor.permute((1, 0))
    print('output permute:',tensor.shape)
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.cpu().data.numpy() * 255.).astype(np.uint8)#将0~1之间的像素值，转化成0~255
    Image.fromarray(im).save(name + '.jpg')


#features各层输出
if False:
    layerCount=30
    for i in range(layerCount+1):
        
        print('------features%d------'%i)
        current_layer = model.features[i]
        if 'Conv2d' in str(current_layer):
            print('input:',images.shape)
            print('layer:',current_layer)
            layer_out = current_layer(images.to(device))
            print('output',layer_out.shape)
            save_img_gray(layer_out, 'features_gray'+str(i))
            images=layer_out

#features各层输出
if True:
    layerCount=len(model.features)
    for i in range(layerCount+1):

        print('------features%d------'%i)
        current_layer = model.features[i]
        nextLayer='NULL'
        if i+2<layerCount+1:
            nextLayer = model.features[i+2]
        print('input:',images.shape)

        print('layer:',current_layer)
        print('nextLayer:',nextLayer)
        layer_out = current_layer(images.to(device))
        print('output',layer_out.shape)
        #save_img(layer_out, 'features'+str(i))
        images = layer_out #下一层网络的输入是上一层网络的输出
        if 'Conv2d' in str(current_layer):

            if 'Conv2d' in str(nextLayer):
                temp_img = images.cpu().detach().numpy()[0]
                temp_img = np.sum(temp_img, 0)
                plt.imshow(temp_img)
                plt.savefig('feature_heatmap_all'+str(i)+'.jpg')
                #cv2.imwrite('feature_heatmap_all'+str(i)+'.jpg', temp_img)
                print(temp_img.shape)
            elif 'MaxPool2d' in str(nextLayer):
                temp_img = images.cpu().detach().numpy()[0]
                temp_img = np.max(temp_img, 0)
                #plt.imshow(temp_img)
                plt.savefig('feature_heatmap_all'+str(i)+'.jpg')
                print(temp_img.shape)

    #将features最后一层输出的数据[1,16,5,5]转化成[1,400]
    images = model.avgpool(images.to(device))
    images = torch.flatten(images.to(device),1)
    #classifier各层输出
    for i in range(5):
        
        current_layer = model.classifier[i]
        print('------classifier%s------'%current_layer)
        print('input:',images.shape)

        print('layer:',current_layer)
        layer_out = current_layer(images.to(device))
        print('output',layer_out.shape)
        
        images = layer_out
        
        print(images.shape)
        #temp_img = images.cpu().detach().numpy()
        save_img_linear(images,'classifier_gray_'+str(i)+'.jpg')