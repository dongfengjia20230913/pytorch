import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import AlexNet
from dataLoader import getAnimalDataloader

import numpy as np
from torch.optim import lr_scheduler


class AlexTrain: 
    #定义基本属性 
    dataType = ''#['Mnist'] or ['Cifar']
    dataRoot = "../datas"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataImgChannel = 3
    EPOCH = 500   #训练总轮数
    #定义构造方法 
    def __init__(self,dataType,dataRoot,dataImgChannel): 
        self.dataType = dataType
        self.dataRoot = dataRoot
        self.dataImgChannel = dataImgChannel



    #初始化网络
    def getLenet(self):

        #getdataload
        #use already define Lenet
        net = AlexNet.alexnet(True,num_classes=6).to(self.device)
        
        loss_fuc = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(net.classifier.parameters(),lr = 0.001,weight_decay = 0.005) 
        return net,loss_fuc,optimizer

    #开始训练
    def train_net(self,net,loss_fuc,optimizer):
        #getdataloader
        train_loader,test_loader= getAnimalDataloader(227)
        #Star train
        
        adjust_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)#定义学习率衰减函数
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
                #print(i)
                if iteration%100==0:#每100次Iteration，测试一次测试数据
                    lr = optimizer.param_groups[0]["lr"]#get current lr
                    #print(lr)
                    print('###iteration[:%d],[Epoch:%d],[Lr:%.08f] train loss: %.03f' % (iteration,epoch + 1, lr, sum_loss / 100))
                    #用网络测试验证数据
                    #保存模型
                    sum_loss = 0.0
                    
            #保存模型
            adjust_lr_scheduler.step()#更新学习率
            self.test_net(self.device,test_loader,net)
            self.save_model(net,optimizer,epoch+1)

    def save_model(self,model,optimizer,epoch):
        print('save_modelt:')
        path = '../models/'+'alexNet_'+self.dataType+'%d.pth'%(epoch)
        print('save net:',path)
        torch.save(model.state_dict(),path)




    #测试验证数据
    def test_net(self,device,test_loader,net):
        correct = 0
        total = 0
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
    def load_model_and_test(self,path):

        #加载模型
        model = self.load_model(path)
        #加载数据
        animaldataloader = dataloader()
        _,test_loader=getAnimalDataloader(227)
        #开始测试
        
        self.test_net(self.device,test_loader,model)
        #print(model)
    


 #训练和测试Mnist
if False:
    print('------train and test Mnist------') 
    dataType='Mnist'
    dataRoot = "../datas"
    dataImgChannel = 1
    ##初始化
    lenetTrain = AlexTrain(dataType,dataRoot,dataImgChannel)
    #获取网络
    net,loss_fuc,optimizer = lenetTrain.getLenet()
    #开始训练
    lenetTrain.train_net(net,loss_fuc,optimizer)

    path= '../models/lenet_Mnist3.pth'
    lenetTrain.load_model_and_test(path)#acc 96%



 #训练和测试Cifar 10
if True:
    print('------train and test Cifar_10------') 
    dataType='Cifar'
    dataRoot = "../datas/cifar"
    ##初始化
    alexNetTrain = AlexTrain(dataType,dataRoot,3)
    #获取网络
    net,loss_fuc,optimizer = alexNetTrain.getLenet()
    #开始训练
    alexNetTrain.train_net(net,loss_fuc,optimizer)

    #path= '../models/alexNet_Cifar5.pth'
    #alexNetTrain.load_model_and_test(path)#acc 67%


