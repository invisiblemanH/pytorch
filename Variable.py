# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:20:48 2019

@author: Administrator
"""

import torch as t
from torch.autograd import Variable as V
x=V(t.Tensor([1]),requires_grad=True)
w=V(t.Tensor([2]),requires_grad=True)
b=V(t.Tensor([3]),requires_grad=True)
y=w*x+b
y.backward()
print(x.grad)
print(w.grad)
print(b.grad)

x=V(t.randn(3),requires_grad=True)
y=x**2
#当结果是多维向量时，不能直接写成y.backward(),传入与y的shape相对应
y.backward(t.Tensor([1.0,1.0,1.0]))
print(x.grad)