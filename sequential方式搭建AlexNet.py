# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:38:12 2019

@author: Administrator
"""

import torch as t
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features=nn.Sequential(
                #conv1
                #输入通道数a（int），输出通道数b(int)。卷积核的个数为b组a。a个卷积核卷积再相加。
                #卷积核的尺寸kernel_size（int or tuple）,卷积步长stride
                #padding(int or tuplepadding 的操作就是在图像块的周围加上格子, 
                #其中padding补0 的策略是四周都补,如果padding=1,那么就会在原来输入层的基础上,
                #上下左右各补一行,如果padding=(1,1)中第一个参数表示在高度上面的padding,第二个参数表示在宽度上面的padding.
                #dilation(int or tuple)控制kernel点（卷积核点）的间距
                #group(int)groups 决定了将原输入分为几组，而每组channel重用几次，由out_channels/groups计算得到，
                #这也说明了为什么需要 groups能供被 out_channels与in_channels整除。
                #bias(bool)是否添加偏置
                nn.Conv2d(3,96,kernel_size=11,stride=4),
                nn.ReLU(inplace=True),#对原变量进行覆盖
                nn.MaxPool2d(kernel_size=3,stride=2),
                nn.ReLU(inplace=True),
                
                #conv2
                nn.Conv2d(96,256,kernel_size=5,padding=2),
                nn.ReLU(inplace=True),#对原变量进行覆盖
                nn.MaxPool2d(kernel_size=3,stride=2),
                nn.ReLU(inplace=True),
                
                #conv3
                nn.Conv2d(256,384,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),#对原变量进行覆盖
                
                #conv4
                nn.Conv2d(384,384,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                
                #conv5
                nn.Conv2d(384,256,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2)
                
                )
        self.classifier=nn.Sequential(
                #fc6
                nn.Linear(256*6*6,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                
                #fc7
                nn.Linear(4096,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                
                #fc8
                nn.Linear(4096,1000)
                )

    def forward(self, x):
        #输入x->conv1->relu->2*2窗口的最大池化
        x=self.features(x)
        #view函数将张量x变形成一维向量的形式，总特征数不变，为全连接层做准备
        x=x.view(x.size(0),256*6*6)
        x=self.classifier(x)
        return x

print("Method 2:")
model2 = AlexNet()
print(model2)