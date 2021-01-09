import _init_paths

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Lenet.leNet import LeNet 

import numpy as np
from torch.optim import lr_scheduler

from net_utils import save_model, load_model
class LeNetTrain: 
    #定义基本属性 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCH = 500   #训练总轮数
    #定义构造方法 
    def __init__(self,dataType,dataRoot,dataImgChannel): 
        self.dataType = dataType
        self.dataRoot = dataRoot
        self.dataImgChannel = dataImgChannel
        self.step_size=80

    #定义数据加载器
    def getDataLoader(self):
        #use gpu to load and train data

        #define transform：
        #1:resize MNIST data to 32x32, so adapter to LeNet struct
        #2：transform PIL.Image to  torch.FloatTensor
        resize = 32
        normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        if self.dataImgChannel==1:
            normalize = transforms.Normalize((0.5), (0.5))
        elif self.dataImgChannel==3:
            normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

        transform = transforms.Compose(
            [transforms.Resize(size=(resize, resize)),
             transforms.ToTensor(),#将图片转换成(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之间
             normalize])#对每个通道通过(image-mean)/std将数据转换到[-1,1]之间
        if dataType=='Mnist':
            train_data = torchvision.datasets.MNIST(root=self.dataRoot,    #data dir 
                                                    train=True,               #it is train data
                                                    transform=transform,      #use defined transform
                                                    download=True)            #use local data
            test_data = torchvision.datasets.MNIST(root=self.dataRoot,
                                                    train=False,
                                                    transform=transform,
                                                    download=True)
        elif dataType=='Cifar':
             train_data = torchvision.datasets.CIFAR10(root='./datas/cifar', train=True, download=True, transform=transform)
             test_data = torchvision.datasets.CIFAR10(root='./datas/cifar', train=False, download=True, transform=transform)
        else:
            print('dataType not in[Mnist,Cifar]')
            return

        #define DataLoder,and shuffle data
        train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size =1000 , shuffle = True)
        test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 1000, shuffle = False)


        return train_loader,test_loader

    #初始化网络
    def getLenet(self):

        #getdataload
        #use already define Lenet
        net = LeNet(self.dataImgChannel).to(self.device)

        loss_fuc = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(net.parameters(),lr = 0.01, weight_decay = 0.005) 
        return net,loss_fuc,optimizer



    #开始训练
    def train_net(self,net,loss_fuc,optimizer):
        #getdataloader
        train_loader,test_loader= self.getDataLoader()
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

                sum_loss += loss.item()
                if iteration%20==0:#每100次Iteration，测试一次测试数据
                    lr = optimizer.param_groups[0]["lr"]#get current lr
                    #print(lr)
                    print('###iteration[:%d],[Epoch:%d],[Lr:%.08f] train sum_loss: [%.03f] -> avg loss[%.03f]' % (iteration,epoch, lr, sum_loss, sum_loss / 20))
                    #用网络测试验证数据
                    #保存模型
                    sum_loss = 0.0
            adjust_lr_scheduler.step()#更新学习率
            self.test_net(self.device,test_loader,net)
            if epoch%30==0 or epoch==self.EPOCH:
                dummy_input = torch.randn(1, 1, 32, 32).to(self.device)   # 生成张量
                save_model(net, optimizer, epoch+1, 'lenet', 'mnist', dummy_input)


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
                #统计test_loader中图片的总个数
                total += labels.size(0) 
                #统计test_loader中 正确分类的个数
                correct += (predicted_label == labels).sum()
        print('test data avg accuracy：%d%%' % ((100 * correct // total)))


    #测试验证集
    def load_model_and_test(self,path):

        #加载模型
        net = LeNet(self.dataImgChannel).to(self.device)
        model = load_model(net, path)
        #加载数据
        _,test_loader= self.getDataLoader()
        #开始测试
        self.test_net(self.device,test_loader,model)
        #print(model)



 #训练和测试Mnist
if True:
    print('------train and test Mnist------') 
    dataType='Mnist'
    dataRoot = "./datas"
    dataImgChannel = 1
    ##初始化
    lenetTrain = LeNetTrain(dataType,dataRoot,dataImgChannel)
    #获取网络
    net,loss_fuc,optimizer = lenetTrain.getLenet()
    #torch.save(net, './Lenet.pth')
    #开始训练
    #lenetTrain.train_net(net,loss_fuc,optimizer)

    path= './models/lenet_mnist/mnist_1.pth'
    lenetTrain.load_model_and_test(path)#acc 96%



 #训练和测试Cifar 10
if False:
    print('------train and test Cifar 10------') 
    dataType='Cifar'
    dataRoot = "../datas/cifar"
    dataImgChannel = 3
    ##初始化
    lenetTrain = LeNetTrain(dataType,dataRoot,dataImgChannel)
    #获取网络
    net,loss_fuc,optimizer = lenetTrain.getLenet()
    #开始训练
    lenetTrain.train_net(net,loss_fuc,optimizer)

    path= './models/lenet_Cifar500.pth'
    lenetTrain.load_model_and_test(path)#acc 67%

