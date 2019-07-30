# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:58:21 2019

@author: Administrator
"""

import torch as t
class myDataset(t.nn.data.Dataset):
    def __init__(self,dataSource):
        self.dataSource=dataSource
    
    def __getitem__(self,index):
        element=self.dataSource[index]
        return element
    
    def __len__(self):
        return len(self.dataSource)
    
train_data=myDataset(dataSource)

class ShipDataset(Dataset):
    #root:图像存放地址根路径。augment:是否需要图像增强
    def __init__(self,root,augment=None):
        #这个list存放所有图像的地址
        self.imge_files=np.array([x.path for x in os.scandir(root) if 
                                  x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        self.augment=augment#
        
    def __getitem__(self,index):
        #读取图像数据并返回
        #这里的open_image是读取图像数据，可以用PIL,opencv等库进行读取
        if self.augment:
            image=open_image(self.imge_files[index])    
            image=self.augment(image)  #在这里进行了图像增强
            return to_tensor(image)    #将读取到的图像变成tensor再传出,在Pytorch中得到的图像必须是tensor
        else:
            #如果不进行增强，直接读取图像数据并返回
            #这里的open_image是读取图像的函数，就可以用PIL,opencv等库存进行读取
            return to_tensor(open_image(self.imge_files[index]))
            
    def __len__(self):
        #返回图像的数量
        return len(self.imge_files)
    
    
#之前所说的Dataset类是读入数据集数据并且对读入的数据进行了索引。但是光有这个功能是不够用的，在实际的加载数据集的过程中，
#我们的数据量往往都很大，对此我们还需要一下几个功能：
#可以分批次读取：batch-size
#可以对数据进行随机读取，可以对数据进行洗牌操作(shuffling)，打乱数据集内数据分布的顺序
#可以并行加载数据(利用多核处理器加快载入数据的效率)
#这时候就需要Dataloader类了，Dataloader这个类并不需要我们自己设计代码，我们只需要利用DataLoader类读取我们设计好的ShipDataset即可：
#    batch_size(int, optional): 每个batch有多少个样本
#    shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
#    sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
#    batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
#    num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
#    collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数
#    pin_memory (bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.

#    drop_last (bool, optional): 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
#    如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。

#    timeout(numeric, optional): 如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。这个numeric应总是大于等于0。默认为0
#    worker_init_fn (callable, optional): 每个worker初始化函数 If not None, this will be called on each
#    worker subprocess with the worker id (an int in [0, num_workers - 1]) as
#    input, after seeding and before data loading. (default: None) 

dataset = MyDataset()
dataloader = DataLoader(dataset)
num_epoches = 100
for epoch in range(num_epoches):
    for img, label in dataloader:
        ....
















