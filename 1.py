#
import torch as t
from torch.autograd import Variable as V
batch_n=100
hidden_layer=100
input_data=1000
output_data=10
#Y=randn(m,n)
#生成m×n随机矩阵，其元素服从均值为0，方差为1的标准正态分布。
#因此randn函数常用来产生高斯白噪声信号
x=V(t.randn(batch_n,input_data),requires_grad=False)
y=V(t.randn(batch_n,output_data),requires_grad=False)
w1=V(t.randn(input_data,hidden_layer),requires_grad=True)
w2=V(t.randn(hidden_layer,output_data),requires_grad=True)

epoch_n=20
learning_rate=1e-6
for epoch in range(epoch_n):
    #torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
    #torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
    #clamp表示夹紧，夹住的意思，torch.clamp(input,min,max,out=None)-> Tensor
    #将input中的元素限制在[min,max]范围内并返回一个Tensor
    y_pred=x.mm(w1).clamp(min=0).mm(w2)
    loss=(y_pred-y).pow(2).sum()
    print("Epoch: {} ,Loss:{:.4f}".format(epoch,loss))
    loss.backward()
    
    w1.data-=learning_rate*w1.grad.data
    w2.data-=learning_rate*w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()