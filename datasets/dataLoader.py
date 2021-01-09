
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
import seaborn as sb
from collections import OrderedDict
import torchvision
from classifier.smokeDataset import SmokeData



def getAnimalDataloader(dataresize=64):
        train_dir = '../datas/animal/train'
        test_dir = '../datas/animal/test'
        train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                               transforms.Resize((dataresize,dataresize)),
                                               transforms.RandomHorizontalFlip(0.5), 
                                               transforms.ColorJitter(brightness=[0.8,1.3], contrast=[0.8,1.3], saturation=[0.8,1.3], hue=0.2),
                                               transforms.ToTensor(), 
                                               transforms.Normalize((0.422, 0.394, 0.347),
                                                                    (0.245, 0.238, 0.232))])

        test_transforms = transforms.Compose([transforms.Resize((dataresize,dataresize)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.422, 0.394, 0.347),
                                                                         (0.245, 0.238, 0.232))])
                                                                         
        # 使用预处理格式加载图像
        train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
        valid_data = datasets.ImageFolder(test_dir,transform = test_transforms)

        # 创建三个加载器，分别为训练，验证，测试，将训练集的batch大小设为64，即每次加载器向网络输送64张图片
        #shuffle 随机打乱，网络更容易学习不同的特征，更容易收敛
        print('load dataset......')
        trainloader = torch.utils.data.DataLoader(train_data,batch_size = 64,shuffle = True)
        validloader = torch.utils.data.DataLoader(valid_data,batch_size = 64)

        return trainloader,validloader


def getSmokeDataloader(train_dir, test_dir, dataresize=64):
        train_transforms = transforms.Compose([
                                               transforms.RandomRotation(20),
                                               transforms.Resize((dataresize,dataresize)),
                                               transforms.RandomHorizontalFlip(0.5), 
                                               #transforms.ColorJitter(brightness=[0.8,1.3], contrast=[0.8,1.3], saturation=[0.8,1.3], hue=0.2),
                                               transforms.ToTensor(), 
                                               transforms.Normalize((0.479, 0.385, 0.352),
                                                                    (0.194, 0.171, 0.165))])

        test_transforms = transforms.Compose([
                                            transforms.Resize((dataresize,dataresize)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.479, 0.385, 0.352),
                                                            (0.194, 0.171, 0.165))])
        tain_smoke_data = SmokeData(train_dir, transform = train_transforms)
        test_smoke_data = SmokeData(test_dir, transform = test_transforms)
        # 使用预处理格式加载图像
        #train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
        #valid_data = datasets.ImageFolder(test_dir,transform = test_transforms)

        # 创建三个加载器，分别为训练，验证，测试，将训练集的batch大小设为64，即每次加载器向网络输送64张图片
        #shuffle 随机打乱，网络更容易学习不同的特征，更容易收敛
        print('load dataset......')
        trainloader = torch.utils.data.DataLoader(tain_smoke_data,batch_size = 64,shuffle = True)
        validloader = torch.utils.data.DataLoader(test_smoke_data,batch_size = 64)

        return trainloader,validloader


def getCifar10DataLoader(resize=32):

    normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    transform = transforms.Compose([transforms.Resize(size=(resize, resize)),
                                    transforms.ToTensor(),
                                    normalize])

    train_data = torchvision.datasets.CIFAR10(root='../datas/cifar', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='../datas/cifar', train=False, download=True, transform=transform)
    #define DataLoder,and shuffle data
    train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size =32 ,shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_data,batch_size = 32,shuffle = True)
    return train_loader,test_loader



def getMnistDataLoader(resize=32):

    normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

    normalize = transforms.Normalize((0.5), (0.5))
    transform = transforms.Compose(
        [transforms.Resize(size=(resize, resize)),
         transforms.ToTensor(),
         normalize])
   
    train_data = torchvision.datasets.CIFAR10(root='../datas/cifar', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='../datas/cifar', train=False, download=True, transform=transform)


    train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size =3200 ,shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_data,batch_size = 1000,shuffle = True)


    return train_loader,test_loader
    
    
def getFireDataloader(dataresize=64,batchsize=64):
        train_dir = '../datas/fire/train'
        test_dir = '../datas/fire/test'
        train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                               transforms.Resize((dataresize,dataresize)),
                                               transforms.RandomHorizontalFlip(0.5), 
                                               #transforms.ColorJitter(brightness=[0.8,1.3], contrast=[0.8,1.3], saturation=[0.8,1.3], hue=0.2),
                                               transforms.ToTensor(), 
                                               transforms.Normalize((0.602, 0.512, 0.455),
                                                                    (0.202, 0.197, 0.181))])

        test_transforms = transforms.Compose([transforms.Resize((dataresize,dataresize)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.602, 0.512, 0.455),
                                                                    (0.202, 0.197, 0.181))])
                                                                         
        # 使用预处理格式加载图像
        train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
        valid_data = datasets.ImageFolder(test_dir,transform = test_transforms)

        # 创建三个加载器，分别为训练，验证，测试，将训练集的batch大小设为64，即每次加载器向网络输送64张图片
        #shuffle 随机打乱，网络更容易学习不同的特征，更容易收敛
        print('load dataset......')
        trainloader = torch.utils.data.DataLoader(train_data,batch_size = batchsize,shuffle = True)
        validloader = torch.utils.data.DataLoader(valid_data,batch_size = batchsize)

        return trainloader,validloader