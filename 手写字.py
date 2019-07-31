# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:47:07 2019

@author: Administrator
"""
import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import os

batch_size=64          #分批训练数据，每批训练数据量
learning_rate=1e-2     #学习率
num_epoches=20         #训练次数

class simpleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1=nn.Sequential(
                nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True)
                )
        self.layer2=nn.Sequential(
                nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True)
                )
        self.layer3=nn.Sequential(
                nn.Linear(n_hidden_2,out_dim)
                )
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        y=self.layer3(x)
        return x
    


data_tf=transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize([0.5],[0.5])
                ]
        )
#下载数据集
train_dataset=datasets.MNIST(
        root='./data',train=True,transform=data_tf,download=True
        )
test_dataset=datasets.MNIST(
        root='./data',train=False,transform=data_tf
        )

#改接口主要用来将自定义的数据读取接口的输出或者pytorch已有的数据读取接口的输入
#按照batch size封装成Tensor,后续只需要再包装成Variable即可作为模型的输入
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

model=simpleNet(28*28,300,100,10) #将28*28像素的图像二维数据展开成一维向量作为肾经网络的输入
criterion=nn.CrossEntropyLoss()   #多分类用的交叉熵损失函数
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

#常用优化方法有
#1.Stochastic Gradient Descent (SGD)
#2.Momentum
#3.AdaGrad
#4.RMSProp
#5.Adam (momentum+adaGrad)   效果较好

if torch.cuda.is_available():#是否可用GPU计算
    model=model.cuda()       #可以使用GPU计算的模型

model.eval()
eval_loss=0
eval_acc=0

for epoch in range(num_epoches):
    print('epoch{}'.format(epoch+1))
    print('*'*10)
    running_loss=0.0
    running_acc=0.0  #准确度
    #训练
    for i,data in enumerate(train_loader,1):
        img,label=data
        img=img.view(img.size(0),-1)        #view()修改维度，即修改shape
        #判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out=model(img)
        loss=criterion(out,label)#计算预测结果out和实际结果label的误差损失，(out为每个预测分类的概率)
        running_loss+=loss.item()*label.size(0)  #先计算总的损失
        _,pred=torch.max(out,1)  #torch.max返回指定维度（1）中的最大值和相应序号，所以pred为预测的分类
        num_correct=(pred==label).sum()
        running_acc+=num_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
        epoch+1,running_loss/(len(train_dataset)),running_acc/len(train_dataset)
    )) 
    #测试
    model.eval()        #eval()时，模型会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0),-1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out,label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc/len(test_dataset)))

































