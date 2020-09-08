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

class classifier():


    def __init__(self):
         # 设置数据目录
        self.train_dir = 'smoke_clas/train'
        self.valid_dir = 'smoke_clas/valid'
        self.test_dir = 'smoke_clas/test'
    #使用dataloader数据，验证模型的准确率
    def accuracy_valid(self, model,dataloader):
        correct = 0
        total = 0
        model.cuda() # 将模型放入GPU计算，能极大加快运算速度
        with torch.no_grad(): # 使用验证集时关闭梯度计算
            for data in dataloader:
               
                images,labels = data
                images,labels = images.to('cuda'),labels.to('cuda')

                outputs = model(images)
                _, predicted = torch.max(outputs.data,1) 
                # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
                total += labels.size(0)
                
                correct += (predicted == labels).sum().item()
                #将预测及标签两相同大小张量逐一比较各相同元素的个数
        print('the accuracy is {:.4f}'.format(correct/total))

    def accuracy_test(self, model,dataloader):
        correct = 0
        total = 0
        model.cuda() # 将模型放入GPU计算，能极大加快运算速度
        with torch.no_grad(): # 使用验证集时关闭梯度计算
            for data in dataloader:
               
                images,labels = data
                images,labels = images.to('cuda'),labels.to('cuda')
                print('label:',labels)

                outputs = model(images)
                probs= []
                classes = []
                #print(outputs.data)
                a = outputs[0]     # 返回TOPK函数截取的排名前列的结果列表a
                b = outputs[1].tolist() #返回TOPK函数截取的排名前列的概率索引列表b
                print('----------------')
                print('a:',a[0])
                print('b:',b[0])
                for i in a:
                   print(torch.exp(i).tolist())
                   probs.append(torch.exp(i).tolist())  #将结果转化为实际概率
                for n in b:
                   classes.append(str(n+1))      # 将索引转化为实际编号
                print(classes)
                print(probs)
                _, predicted = torch.max(outputs.data,1) 
                print('label:',labels)
                print('predicted:',predicted)
                #print(predicted)



    def deep_learning(self, model,trainloader,epochs,print_every,criterion,optimizer,device,validloader):
        epochs = epochs #设置学习次数
        print_every = print_every
        steps = 0
        model.to(device)
        
        for e in range(epochs):
            running_loss = 0
            for ii , (inputs,labels) in enumerate(trainloader):
                steps += 1
                #inputs表示输入的数据，labels表示每个数据对应的真实分类标签
                inputs,labels = inputs.to(device),labels.to(device)
                optimizer.zero_grad() # 优化器梯度清零
                
                # 前馈及反馈
                outputs = model(inputs) #前馈过程
                loss = criterion(outputs,labels) # 计算误差
                loss.backward() #后馈过程
                optimizer.step() #优化器更新参数
                #计算迭代误差和
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    #test the accuracy
                   
                    #打印中间迭代结果，和平均误差
                    print('EPOCHS : {}/{}'.format(e+1,epochs),
                          'Loss : {:.4f}'.format(running_loss/print_every))
                    self.accuracy_valid(model,validloader)
                    torch.save(model.state_dict(), 'models/'+'smoke_clas_'+str(steps)+'.pth')

    def data_pre(self):

        image_size = 224
        train_transforms = transforms.Compose([transforms.RandomRotation(30),#随机旋转30
                                               #transforms.RandomResizedCrop(224),#对人脸的场景不能选择随机剪切，因为两个分类的只有小的局部有差别
                                               transforms.Resize((image_size,image_size)),
                                               transforms.RandomHorizontalFlip(), # 水平翻转
                                               transforms.ToTensor(), # 对图像进行张量化，以便神经网络处理
                                               transforms.Normalize((0.485,0.456,0.406),
                                                                    (0.229,0.224,0.225))])

        valid_transforms = transforms.Compose([transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485,0.456,0.406],
                                                                         [0.229,0.224,0.225])])
                                                                         
        # 使用预处理格式加载图像
        train_data = datasets.ImageFolder(self.train_dir,transform = train_transforms)
        valid_data = datasets.ImageFolder(self.valid_dir,transform = valid_transforms)

        # 创建三个加载器，分别为训练，验证，测试，将训练集的batch大小设为64，即每次加载器向网络输送64张图片
        #shuffle 随机打乱，网络更容易学习不同的特征，更容易收敛
        print('load dataset......')
        trainloader = torch.utils.data.DataLoader(train_data,batch_size = 64,shuffle = True)
        validloader = torch.utils.data.DataLoader(valid_data,batch_size = 20)


        print('load models......')
        #使用models下载vgg16神经网络，是CNN卷积网络中的一种，比较小，也好用
        #表示只导入网络结构，不导入参数
        fmodel = models.vgg16(pretrained = False)

        # 定义一个新的classifier，其中包含3个全连接隐藏层，层之间使用ReLU函数进行激活，最后输出使用LogSoftMax函数，因为这是一个多分类。
        classifier = nn.Sequential(OrderedDict([
                                    ('fc1',nn.Linear(25088,4096)),
                                     ('relu1',nn.ReLU()),
                                     ('fc2',nn.Linear(4096,1000)),
                                     ('relu2',nn.ReLU()),
                                     ('fc3',nn.Linear(1000,2)),
                                     ('output',nn.LogSoftmax(dim=1))
        ]))
        # 替换
        fmodel.classifier = classifier

        #print(fmodel)

        # 使用Negative Log Likelihood Loss作为误差函数

        criterion = nn.NLLLoss()

        # 使用Adam作为优化器，并且只对分类器的参数进行优化，也可以使用SDG，ADAM具有动量优化效果，设置学习速率为0.01，
        optimizer = optim.Adam(fmodel.classifier.parameters(),lr=0.001)
        return fmodel,trainloader,criterion,optimizer,validloader


if __name__ == '__main__':
    demo = classifier()
    fmodel,trainloader,criterion,optimizer,validloader = demo.data_pre()
    print('start train.....')
    demo.deep_learning(fmodel,trainloader,200,100,criterion,optimizer,'cuda',validloader)

