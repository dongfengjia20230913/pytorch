import _init_paths

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from VGG.vgg import vgg16
from datasets.dataLoader import getAnimalDataloader,getSmokeDataloader,getFireDataloader,getCifar10DataLoader
from torchvision import datasets,transforms,models
from collections import OrderedDict
import os
import torch.nn as nn



import numpy as np
from torch.optim import lr_scheduler


class NetTrain: 

    #定义构造方法 
    def __init__(self): 
        print('init NetTrain')
       #定义基本属性 
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.EPOCH = 500   #训练总轮数
        self.step_size=80
        self.input_shape=227
        self.dataType='smoke'

    #初始化网络
    def getNetAndLoader(self):

        #getdataload
        #use already define Lenet
        net = vgg16(pretrained=False,progress=False).to(self.device)
        train_dir = './datas/smoke_clas/train'
        test_dir = './datas/smoke_clas/test'
        train_loader,test_loader=getSmokeDataloader(train_dir, test_dir, self.input_shape)

        #net = models.vgg16(pretrained = True)

        # 定义一个新的classifier，其中包含3个全连接隐藏层，层之间使用ReLU函数进行激活，最后输出使用LogSoftMax函数，因为这是一个多分类。
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
        net.to(self.device)
        loss_fuc = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(net.classifier.parameters(),lr = 0.001) 
        return net,loss_fuc,optimizer,train_loader,test_loader

    #开始训练
    def train_net(self,net,loss_fuc,optimizer,train_loader,test_loader):
        #getdataloader
        #Star train

        adjust_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.1)#定义学习率衰减函数
        print('start train with epoch:',self.EPOCH)
        iteration = 0
        for epoch in range(self.EPOCH):
            sum_loss = 0

            #数据读取
            for i,data in enumerate(train_loader):
                inputs,labels = data
                #有GPU则将数据置入GPU加速
                inputs, labels = inputs.to(self.device), labels.to(self.device)   

                # 梯度清零
                optimizer.zero_grad()

                # 传递损失 + 更新参数
                output = net(inputs)
                loss = loss_fuc(output,labels)
                loss.backward()
                optimizer.step()
                
                iteration = iteration+1

                # print loss every 100 iteration
                sum_loss += loss.item()
                #print(loss.item())
                if iteration%20==0:#每100次Iteration，测试一次测试数据
                    lr = optimizer.param_groups[0]["lr"]#get current lr
                    #print(lr)
                    print('###iteration[:%d],[Epoch:%d],[Lr:%.08f] train sum_loss: [%.03f] -> avg loss[%.03f]' % (iteration,epoch + 1, lr, sum_loss, sum_loss / 20))
                    #用网络测试验证数据
                    #保存模型
                    sum_loss = 0.0
            adjust_lr_scheduler.step()#更新学习率
            self.test_net(self.device,test_loader,net)
            if epoch%30==0 or epoch==self.EPOCH:
                self.save_model(net,optimizer, epoch+1, 'vgg', 'smoke')


    #测试验证数据
    def test_net(self,device,test_loader,net):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                test_inputs, labels = data
                test_inputs, labels = test_inputs.to(self.device), labels.to(self.device)
                outputs_test = net(test_inputs)#输出的是batch_size张图片的10个分类值
                #print('=========outputs_test=============')
                #print(outputs_test.shape)
                predict_value, predicted_label = torch.max(outputs_test.data, 1)  #输出每个数据得分最高的那个分类id
               
                total += labels.size(0) #统计test_loader中图片的总个数
                correct += (predicted_label == labels).sum()  #统计test_loader中 正确分类的个数
                #print('predict_value:',predict_value)
                #print('predicted_label:',predicted_label)
                #print('labels:',labels)
                #print(correct,total,'%d%%' % ((100 * correct // total)))
                #print('\n')
            
        print('test data avg accuracy：%d%%' % ((100 * correct // total)))

if True:
    ##初始化
    netTrain = NetTrain()
    #获取网络
    net,loss_fuc,optimizer,trainLoader,testLoader = netTrain.getNetAndLoader()
    #开始训练
    netTrain.train_net(net,loss_fuc,optimizer,trainLoader,testLoader)

    #path= '../models/alexNet_Cifar5.pth'
   # _,test_loader=getFireDataloader(227,1)
   # alexNetTrain.load_model_and_test(path,test_loader)#acc 67%


