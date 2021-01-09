import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from  AlexNet import  AlexNet
from dataLoader import getAnimalDataloader,getSmokeDataloader,getCifar10DataLoader
from torchvision import datasets,transforms,models
from collections import OrderedDict

import numpy as np
from torch.optim import lr_scheduler


class NetTrain: 
    #定义基本属性 
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    EPOCH = 500   #训练总轮数
    step_size=80
    input_shape=227
    dataType='fire'
    #定义构造方法 
    def __init__(self): 
       print('init NetTrain')



    #初始化网络
    def getNetAndLoader(self):

        #getdataload
        #use already define Lenet
        net = AlexNet(10).to(self.device)
        train_loader,test_loader=getCifar10DataLoader(self.input_shape)

        #net = models.vgg16(pretrained = True)

     
        net.to(self.device)
        loss_fuc = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(net.classifier.parameters(),lr = 0.001) 
        return net,loss_fuc,optimizer,train_loader,test_loader

    #开始训练
    def train_net(self,net,loss_fuc,optimizer,train_loader,test_loader):
        #getdataloader
        #Star train
        
        #adjust_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.1)#定义学习率衰减函数
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
                if iteration%10==0:#每100次Iteration，测试一次测试数据
                    lr = optimizer.param_groups[0]["lr"]#get current lr
                    #print(lr)
                    print('###iteration[:%d],[Epoch:%d],[Lr:%.08f] train loss: %.03f' % (iteration,epoch + 1, lr, sum_loss / 10))
                    #用网络测试验证数据
                    #保存模型
                    sum_loss = 0.0
                    #adjust_lr_scheduler.step()#更新学习率
            if epoch%20==0:
                self.test_net(self.device,test_loader,net)
                self.save_model(net,optimizer,epoch+1)

    def save_model(self,model,optimizer,epoch):
        print('save_modelt:')
        path = '../models/'+'AlexNet_'+self.dataType+'%d.pth'%(epoch)
        print('save net:',path)
        torch.save(model.state_dict(),path)




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

    def load_model(self,path):
        net = AlexNet.alexnet(False,6).to(self.device)
        net.load_state_dict(torch.load(path))
        return net

    #测试验证集
    def load_model_and_test(self,path,test_loader):

        #加载模型
        model = self.load_model(path)
        #加载数据
        animaldataloader = dataloader()
        #开始测试
        
        self.test_net(self.device,test_loader,model)
        #print(model)
    






if True:
    ##初始化
    netTrain = NetTrain()
    #获取网络
    net,loss_fuc,optimizer,trainLoader,testLoader = netTrain.getNetAndLoader()
    #开始训练
    netTrain.train_net(net,loss_fuc,optimizer,trainLoader,testLoader)

    #path= '../models/alexNet_Cifar5.pth'
    #_,test_loader=getFireDataloader(227,1)
    #alexNetTrain.load_model_and_test(path,test_loader)#acc 67%


