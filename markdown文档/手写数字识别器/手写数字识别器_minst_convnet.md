# 我卷卷卷

在这节课中，我们学习了如何使用torchvision加载图像库，如何构建一个简单的卷积神经网络，并了解如何训练这个卷积神经网络
之后，我们还学会了如何对训练好的卷积网络进行分析

本文件是集智学园http://campus.swarma.org 出品的“火炬上的深度学习”第III课的配套源代码


```python
pip install torchvision
```

    Requirement already satisfied: torchvision in c:\users\lxy\appdata\roaming\python\python39\site-packages (0.15.2)
    Requirement already satisfied: requests in d:\download\anaconda3\lib\site-packages (from torchvision) (2.28.1)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in d:\download\anaconda3\lib\site-packages (from torchvision) (9.2.0)
    Requirement already satisfied: torch==2.0.1 in c:\users\lxy\appdata\roaming\python\python39\site-packages (from torchvision) (2.0.1)
    Requirement already satisfied: numpy in d:\download\anaconda3\lib\site-packages (from torchvision) (1.26.3)
    Requirement already satisfied: sympy in d:\download\anaconda3\lib\site-packages (from torch==2.0.1->torchvision) (1.10.1)
    Requirement already satisfied: typing-extensions in d:\download\anaconda3\lib\site-packages (from torch==2.0.1->torchvision) (4.3.0)
    Requirement already satisfied: networkx in d:\download\anaconda3\lib\site-packages (from torch==2.0.1->torchvision) (2.8.4)
    Requirement already satisfied: jinja2 in d:\download\anaconda3\lib\site-packages (from torch==2.0.1->torchvision) (2.11.3)
    Requirement already satisfied: filelock in d:\download\anaconda3\lib\site-packages (from torch==2.0.1->torchvision) (3.6.0)
    Requirement already satisfied: certifi>=2017.4.17 in d:\download\anaconda3\lib\site-packages (from requests->torchvision) (2022.9.14)
    Requirement already satisfied: idna<4,>=2.5 in d:\download\anaconda3\lib\site-packages (from requests->torchvision) (3.3)
    Requirement already satisfied: charset-normalizer<3,>=2 in d:\download\anaconda3\lib\site-packages (from requests->torchvision) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in d:\download\anaconda3\lib\site-packages (from requests->torchvision) (1.26.11)
    Requirement already satisfied: MarkupSafe>=0.23 in d:\download\anaconda3\lib\site-packages (from jinja2->torch==2.0.1->torchvision) (2.0.1)
    Requirement already satisfied: mpmath>=0.19 in d:\download\anaconda3\lib\site-packages (from sympy->torch==2.0.1->torchvision) (1.2.1)
    Note: you may need to restart the kernel to use updated packages.


    WARNING: Ignoring invalid distribution -orch (d:\download\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -orch (d:\download\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -orch (d:\download\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -orch (d:\download\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -orch (d:\download\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -orch (d:\download\anaconda3\lib\site-packages)



```python
# 导入所需要的包，请保证torchvision已经在你的环境中安装好.
# 在Windows需要单独安装torchvision包，在命令行运行pip install torchvision即可
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
```

## 一、加载数据

1. 首先，我们需要学习PyTorch自带的数据加载器，包括dataset，sampler，以及data loader这三个对象组成的套件。
2. 当数据集很小，格式比较规则的时候，数据加载三套件的优势并不明显。但是当数据格式比较特殊，以及数据规模很大（内存无法同时加载所有数据）
的时候，特别是，我们需要用不同的处理器来加载数据的时候，三套件的威力就会显现出来了。它会将数据加载、分布的任务自动完成。
3. 在使用的时候，我们用dataset来装载数据集，用sampler来采样数据集。而对数据集的迭代、循环则主要通过data_loader来完成。
4. 创建一个data_loader就需要一个dataset和一个datasampler，它基本实现的就是利用sampler自动从dataset种采样

本文件是集智AI学园http://campus.swarma.org 出品的“火炬上的深度学习”第III课的配套源代码


```python

# 定义超参数 
image_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 20  #训练的总循环周期
batch_size = 64  #一个撮（批次）的大小，64张图片

# 加载MINIST数据，如果没有下载过，就会在当前路径下新建/data子目录，并把文件存放其中
# MNIST数据是属于torchvision包自带的数据，所以可以直接调用。
# 在调用自己的数据的时候，我们可以用torchvision.datasets.ImageFolder或者torch.utils.data.TensorDataset来加载
train_dataset = dsets.MNIST(root='./data',  #文件存放路径
                            train=True,   #提取训练集
                            transform=transforms.ToTensor(),  #将图像转化为Tensor，在加载数据的时候，就可以对图像做预处理
                            download=True) #当找不到文件的时候，自动下载

# 加载测试数据集
test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# 训练数据集的加载器，自动将数据分割成batch，顺序随机打乱
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

'''我们希望将测试数据分成两部分，一部分作为校验数据，一部分作为测试数据。
校验数据用于检测模型是否过拟合，并调整参数，测试数据检验整个模型的工作'''


# 首先，我们定义下标数组indices，它相当于对所有test_dataset中数据的编码
# 然后定义下标indices_val来表示校验集数据的那些下标，indices_test表示测试集的下标
indices = range(len(test_dataset))
indices_val = indices[:5000]
indices_test = indices[5000:]

# 根据这些下标，构造两个数据集的SubsetRandomSampler采样器，它会对下标进行采样
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

# 根据两个采样器来定义加载器，注意将sampler_val和sampler_test分别赋值给了validation_loader和test_loader
validation_loader = torch.utils.data.DataLoader(dataset =test_dataset,
                                                batch_size = batch_size,
                                                sampler = sampler_val
                                               )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          sampler = sampler_test
                                         )
```


```python
#随便从数据集中读入一张图片，并绘制出来
idx = 100
#dataset支持下标索引，其中提取出来的每一个元素为features，target格式，即属性和标签。[0]表示索引features
muteimg = train_dataset[idx][0].numpy()
#由于一般的图像包含rgb三个通道，而MINST数据集的图像都是灰度的，只有一个通道。因此，我们忽略通道，把图像看作一个灰度矩阵。
#用imshow画图，会将灰度矩阵自动展现为彩色，不同灰度对应不同颜色：从黄到紫

plt.imshow(muteimg[0,...])
print('标签是：',train_dataset[idx][1])
```

    标签是： 5



    
![png](https://github.com/woodpeckerdk/pytorch/blob/main/markdown%E6%96%87%E6%A1%A3/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_6_1.png)
    


## 二、基本的卷积神经网络

### 2.1 构建网络

注：在这里，我们将主要调用PyTorch强大的nn.Module这个类来构建卷积神经网络。我们分成如下这几个步骤：

1. 首先，我们构造ConvNet类，它是对类nn.Module的继承（即nn.Module是父类，ConvNet为子类。如果这些概念不熟悉，请参考面向对象编程）
2. 然后，我们复写__init__，以及forward这两个函数。第一个为构造函数，每当类ConvNet被具体化一个实例的时候，就会调用，forward则是
在运行神经网络正向的时候会被自动调用
3. 自定义自己的方法

另一需要理解的是，ConvNet其实也是一个大容器，它里面有Conv2d，MaxPool2d等组件

本文件是集智AI学园http://campus.swarma.org 出品的“火炬上的深度学习”第III课的配套源代码


```python
#定义卷积神经网络：4和8为人为指定的两个卷积层的厚度（feature map的数量）
depth = [4, 8]
class ConvNet(nn.Module):
    def __init__(self):
        # 该函数在创建一个ConvNet对象的时候，即调用如下语句：net=ConvNet()，就会被调用
        # 首先调用父类相应的构造函数
        super(ConvNet, self).__init__()
        
        # 其次构造ConvNet需要用到的各个神经模块。
        '''注意，定义组件并没有真正搭建这些组件，只是把基本建筑砖块先找好'''
        self.conv1 = nn.Conv2d(1, 4, 5, padding = 2) #定义一个卷积层，输入通道为1，输出通道为4，窗口大小为5，padding为2
        self.pool = nn.MaxPool2d(2, 2) #定义一个Pooling层，一个窗口为2*2的pooling运算
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding = 2) #第二层卷积，输入通道为depth[0], 
                                                                   #输出通道为depth[1]，窗口为5，padding为2
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1] , 512) 
                                                            #一个线性连接层，输入尺寸为最后一层立方体的平铺，输出层512个节点
        self.fc2 = nn.Linear(512, num_classes) #最后一层线性分类单元，输入为512，输出为要做分类的类别数

    def forward(self, x):
        #该函数完成神经网络真正的前向运算，我们会在这里把各个组件进行实际的拼装
        #x的尺寸：(batch_size, image_channels, image_width, image_height)
        x = F.relu(self.conv1(x))  #第一层卷积，激活函数用ReLu，为了防止过拟合
        #x的尺寸：(batch_size, num_filters, image_width, image_height)
        x = self.pool(x) #第二层pooling，将图片变小
        #x的尺寸：(batch_size, depth[0], image_width/2, image_height/2)
        x = F.relu(self.conv2(x)) #第三层又是卷积，窗口为5，输入输出通道分别为depth[0]=4, depth[1]=8
        #x的尺寸：(batch_size, depth[1], image_width/2, image_height/2)
        x = self.pool(x) #第四层pooling，将图片缩小到原大小的1/4
        #x的尺寸：(batch_size, depth[1], image_width/4, image_height/4)
        
        # 将立体的特征图Tensor，压成一个一维的向量
        # view这个函数可以将一个tensor按指定的方式重新排布。
        # 下面这个命令就是要让x按照batch_size * (image_size//4)^2*depth[1]的方式来排布向量
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        #x的尺寸：(batch_size, depth[1]*image_width/4*image_height/4)
        
        x = F.relu(self.fc1(x)) #第五层为全链接，ReLu激活函数
        #x的尺寸：(batch_size, 512)

        x = F.dropout(x, training=self.training) #以默认为0.5的概率对这一层进行dropout操作，为了防止过拟合
        x = self.fc2(x) #全链接
        #x的尺寸：(batch_size, num_classes)
        
        x = F.log_softmax(x, dim = 0) #输出层为log_softmax，即概率对数值log(p(x))。采用log_softmax可以使得后面的交叉熵计算更快
        return x
    
    def retrieve_features(self, x):
        #该函数专门用于提取卷积神经网络的特征图的功能，返回feature_map1, feature_map2为前两层卷积层的特征图
        feature_map1 = F.relu(self.conv1(x)) #完成第一层卷积
        x = self.pool(feature_map1)  # 完成第一层pooling
        feature_map2 = F.relu(self.conv2(x)) #第二层卷积，两层特征图都存储到了feature_map1, feature_map2中
        return (feature_map1, feature_map2)
    
```

### 2.2 训练这个卷积神经网络


```python
def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素

```


```python
net = ConvNet() #新建一个卷积神经网络的实例，此时ConvNet的__init__函数就会被自动调用

criterion = nn.CrossEntropyLoss() #Loss函数的定义，交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #定义优化器，普通的随机梯度下降算法

record = [] #记录准确率等数值的容器
weights = [] #每若干步就记录一次卷积核

#开始训练循环
for epoch in range(num_epochs):
    
    train_rights = [] #记录训练数据集准确率的容器
    
    ''' 下面的enumerate是构造一个枚举器的作用。就是我们在对train_loader做循环迭代的时候，enumerate会自动吐出一个数字指示我们循环了几次
     这个数字就被记录在了batch_idx之中，它就等于0，1，2，……
     train_loader每迭代一次，就会吐出来一对数据data和target，分别对应着一个batch中的手写数字图，以及对应的标签。'''
    
    for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
        data, target = data.clone().requires_grad_(True), target.clone().detach()  #data为一批图像，target为一批标签
        net.train() # 给网络模型做标记，标志说模型正在训练集上训练，
                    #这种区分主要是为了打开关闭net的training标志，从而决定是否运行dropout
            
        output = net(data) #神经网络完成一次前馈的计算过程，得到预测输出output
        loss = criterion(output, target) #将output与标签target比较，计算误差
        optimizer.zero_grad() #清空梯度
        loss.backward() #反向传播
        optimizer.step() #一步随机梯度下降算法
        right = rightness(output, target) #计算准确率所需数值，返回数值为（正确样例数，总样本数）
        train_rights.append(right) #将计算结果装到列表容器train_rights中

    
        if batch_idx % 100 == 0: #每间隔100个batch执行一次打印等操作
            
            net.eval() # 给网络模型做标记，标志说模型在训练集上训练
            val_rights = [] #记录校验数据集准确率的容器
            
            '''开始在校验数据集上做循环，计算校验集上面的准确度'''
            for (data, target) in validation_loader:
                data, target = data.clone().requires_grad_(True), target.clone().detach()
                output = net(data) #完成一次前馈计算过程，得到目前训练得到的模型net在校验数据集上的表现
                right = rightness(output, target) #计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                val_rights.append(right)
            
            # 分别计算在目前已经计算过的测试数据集，以及全部校验集上模型的表现：分类准确率
            #train_r为一个二元组，分别记录目前已经经历过的所有训练集中分类正确的数量和该集合中总的样本数，
            #train_r[0]/train_r[1]就是训练集的分类准确度，同样，val_r[0]/val_r[1]就是校验集上的分类准确度
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            #val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
            #打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
            print(val_r)
            print('训练周期: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.data, 
                100. * train_r[0].numpy() / train_r[1], 
                100. * val_r[0].numpy() / val_r[1]))
            
            #将准确率和权重等数值加载到容器中，以方便后续处理
            record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))
            
            # weights记录了训练周期中所有卷积核的演化过程。net.conv1.weight就提取出了第一层卷积核的权重
            # clone的意思就是将weight.data中的数据做一个拷贝放到列表中，否则当weight.data变化的时候，列表中的每一项数值也会联动
            '''这里使用clone这个函数很重要'''
            weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(), 
                            net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])
            
```

    (tensor(242), 5000)
    训练周期: 0 [0/60000 (0%)]	Loss: 2.305995	训练正确率: 6.25%	校验正确率: 4.84%
    (tensor(1065), 5000)
    训练周期: 0 [6400/60000 (11%)]	Loss: 2.289611	训练正确率: 10.07%	校验正确率: 21.30%
    (tensor(2164), 5000)
    训练周期: 0 [12800/60000 (21%)]	Loss: 2.279700	训练正确率: 13.71%	校验正确率: 43.28%
    (tensor(2708), 5000)
    训练周期: 0 [19200/60000 (32%)]	Loss: 2.255553	训练正确率: 17.67%	校验正确率: 54.16%
    (tensor(3004), 5000)
    训练周期: 0 [25600/60000 (43%)]	Loss: 1.995610	训练正确率: 22.53%	校验正确率: 60.08%
    (tensor(3502), 5000)
    训练周期: 0 [32000/60000 (53%)]	Loss: 1.281992	训练正确率: 28.13%	校验正确率: 70.04%
    (tensor(3815), 5000)
    训练周期: 0 [38400/60000 (64%)]	Loss: 0.823742	训练正确率: 34.42%	校验正确率: 76.30%
    (tensor(3970), 5000)
    训练周期: 0 [44800/60000 (75%)]	Loss: 0.755976	训练正确率: 40.09%	校验正确率: 79.40%
    (tensor(4132), 5000)
    训练周期: 0 [51200/60000 (85%)]	Loss: 0.537609	训练正确率: 44.70%	校验正确率: 82.64%
    (tensor(4146), 5000)
    训练周期: 0 [57600/60000 (96%)]	Loss: 0.491297	训练正确率: 48.62%	校验正确率: 82.92%
    (tensor(4161), 5000)
    训练周期: 1 [0/60000 (0%)]	Loss: 0.443663	训练正确率: 85.94%	校验正确率: 83.22%
    (tensor(4264), 5000)
    训练周期: 1 [6400/60000 (11%)]	Loss: 0.648397	训练正确率: 81.93%	校验正确率: 85.28%
    (tensor(4319), 5000)
    训练周期: 1 [12800/60000 (21%)]	Loss: 0.487611	训练正确率: 82.44%	校验正确率: 86.38%
    (tensor(4385), 5000)
    训练周期: 1 [19200/60000 (32%)]	Loss: 0.619914	训练正确率: 82.91%	校验正确率: 87.70%
    (tensor(4383), 5000)
    训练周期: 1 [25600/60000 (43%)]	Loss: 0.361825	训练正确率: 83.54%	校验正确率: 87.66%
    (tensor(4413), 5000)
    训练周期: 1 [32000/60000 (53%)]	Loss: 0.539680	训练正确率: 83.89%	校验正确率: 88.26%
    (tensor(4469), 5000)
    训练周期: 1 [38400/60000 (64%)]	Loss: 0.448012	训练正确率: 84.40%	校验正确率: 89.38%
    (tensor(4455), 5000)
    训练周期: 1 [44800/60000 (75%)]	Loss: 0.444789	训练正确率: 84.89%	校验正确率: 89.10%
    (tensor(4498), 5000)
    训练周期: 1 [51200/60000 (85%)]	Loss: 0.324185	训练正确率: 85.26%	校验正确率: 89.96%
    (tensor(4487), 5000)
    训练周期: 1 [57600/60000 (96%)]	Loss: 0.530863	训练正确率: 85.67%	校验正确率: 89.74%
    (tensor(4519), 5000)
    训练周期: 2 [0/60000 (0%)]	Loss: 0.285443	训练正确率: 89.06%	校验正确率: 90.38%
    (tensor(4536), 5000)
    训练周期: 2 [6400/60000 (11%)]	Loss: 0.451384	训练正确率: 89.22%	校验正确率: 90.72%
    (tensor(4565), 5000)
    训练周期: 2 [12800/60000 (21%)]	Loss: 0.387425	训练正确率: 89.58%	校验正确率: 91.30%
    (tensor(4593), 5000)
    训练周期: 2 [19200/60000 (32%)]	Loss: 0.401002	训练正确率: 89.61%	校验正确率: 91.86%
    (tensor(4589), 5000)
    训练周期: 2 [25600/60000 (43%)]	Loss: 0.222287	训练正确率: 90.02%	校验正确率: 91.78%
    (tensor(4594), 5000)
    训练周期: 2 [32000/60000 (53%)]	Loss: 0.223981	训练正确率: 90.32%	校验正确率: 91.88%
    (tensor(4624), 5000)
    训练周期: 2 [38400/60000 (64%)]	Loss: 0.186292	训练正确率: 90.48%	校验正确率: 92.48%
    (tensor(4636), 5000)
    训练周期: 2 [44800/60000 (75%)]	Loss: 0.261260	训练正确率: 90.53%	校验正确率: 92.72%
    (tensor(4635), 5000)
    训练周期: 2 [51200/60000 (85%)]	Loss: 0.232542	训练正确率: 90.76%	校验正确率: 92.70%
    (tensor(4640), 5000)
    训练周期: 2 [57600/60000 (96%)]	Loss: 0.368244	训练正确率: 90.95%	校验正确率: 92.80%
    (tensor(4675), 5000)
    训练周期: 3 [0/60000 (0%)]	Loss: 0.413679	训练正确率: 89.06%	校验正确率: 93.50%
    (tensor(4671), 5000)
    训练周期: 3 [6400/60000 (11%)]	Loss: 0.224588	训练正确率: 92.33%	校验正确率: 93.42%
    (tensor(4652), 5000)
    训练周期: 3 [12800/60000 (21%)]	Loss: 0.419229	训练正确率: 92.12%	校验正确率: 93.04%
    (tensor(4679), 5000)
    训练周期: 3 [19200/60000 (32%)]	Loss: 0.260473	训练正确率: 92.31%	校验正确率: 93.58%
    (tensor(4704), 5000)
    训练周期: 3 [25600/60000 (43%)]	Loss: 0.184721	训练正确率: 92.45%	校验正确率: 94.08%
    (tensor(4705), 5000)
    训练周期: 3 [32000/60000 (53%)]	Loss: 0.175111	训练正确率: 92.56%	校验正确率: 94.10%
    (tensor(4688), 5000)
    训练周期: 3 [38400/60000 (64%)]	Loss: 0.210146	训练正确率: 92.74%	校验正确率: 93.76%
    (tensor(4703), 5000)
    训练周期: 3 [44800/60000 (75%)]	Loss: 0.215836	训练正确率: 92.90%	校验正确率: 94.06%
    (tensor(4708), 5000)
    训练周期: 3 [51200/60000 (85%)]	Loss: 0.156764	训练正确率: 93.04%	校验正确率: 94.16%
    (tensor(4731), 5000)
    训练周期: 3 [57600/60000 (96%)]	Loss: 0.241155	训练正确率: 93.04%	校验正确率: 94.62%
    (tensor(4718), 5000)
    训练周期: 4 [0/60000 (0%)]	Loss: 0.178013	训练正确率: 96.88%	校验正确率: 94.36%
    (tensor(4735), 5000)
    训练周期: 4 [6400/60000 (11%)]	Loss: 0.162558	训练正确率: 94.20%	校验正确率: 94.70%
    (tensor(4721), 5000)
    训练周期: 4 [12800/60000 (21%)]	Loss: 0.234553	训练正确率: 94.12%	校验正确率: 94.42%
    (tensor(4719), 5000)
    训练周期: 4 [19200/60000 (32%)]	Loss: 0.137637	训练正确率: 94.09%	校验正确率: 94.38%
    (tensor(4732), 5000)
    训练周期: 4 [25600/60000 (43%)]	Loss: 0.156789	训练正确率: 93.98%	校验正确率: 94.64%
    (tensor(4748), 5000)
    训练周期: 4 [32000/60000 (53%)]	Loss: 0.089083	训练正确率: 94.02%	校验正确率: 94.96%
    (tensor(4748), 5000)
    训练周期: 4 [38400/60000 (64%)]	Loss: 0.237000	训练正确率: 94.04%	校验正确率: 94.96%
    (tensor(4764), 5000)
    训练周期: 4 [44800/60000 (75%)]	Loss: 0.199436	训练正确率: 94.08%	校验正确率: 95.28%
    (tensor(4768), 5000)
    训练周期: 4 [51200/60000 (85%)]	Loss: 0.236734	训练正确率: 94.15%	校验正确率: 95.36%
    (tensor(4758), 5000)
    训练周期: 4 [57600/60000 (96%)]	Loss: 0.203994	训练正确率: 94.19%	校验正确率: 95.16%
    (tensor(4752), 5000)
    训练周期: 5 [0/60000 (0%)]	Loss: 0.114555	训练正确率: 95.31%	校验正确率: 95.04%
    (tensor(4750), 5000)
    训练周期: 5 [6400/60000 (11%)]	Loss: 0.232319	训练正确率: 94.45%	校验正确率: 95.00%
    (tensor(4740), 5000)
    训练周期: 5 [12800/60000 (21%)]	Loss: 0.169075	训练正确率: 94.78%	校验正确率: 94.80%
    (tensor(4743), 5000)
    训练周期: 5 [19200/60000 (32%)]	Loss: 0.255530	训练正确率: 94.65%	校验正确率: 94.86%
    (tensor(4761), 5000)
    训练周期: 5 [25600/60000 (43%)]	Loss: 0.096696	训练正确率: 94.81%	校验正确率: 95.22%
    (tensor(4744), 5000)
    训练周期: 5 [32000/60000 (53%)]	Loss: 0.139265	训练正确率: 94.80%	校验正确率: 94.88%
    (tensor(4783), 5000)
    训练周期: 5 [38400/60000 (64%)]	Loss: 0.213286	训练正确率: 94.84%	校验正确率: 95.66%
    (tensor(4776), 5000)
    训练周期: 5 [44800/60000 (75%)]	Loss: 0.083536	训练正确率: 94.80%	校验正确率: 95.52%
    (tensor(4773), 5000)
    训练周期: 5 [51200/60000 (85%)]	Loss: 0.136396	训练正确率: 94.89%	校验正确率: 95.46%
    (tensor(4785), 5000)
    训练周期: 5 [57600/60000 (96%)]	Loss: 0.093060	训练正确率: 94.95%	校验正确率: 95.70%
    (tensor(4785), 5000)
    训练周期: 6 [0/60000 (0%)]	Loss: 0.102342	训练正确率: 96.88%	校验正确率: 95.70%
    (tensor(4780), 5000)
    训练周期: 6 [6400/60000 (11%)]	Loss: 0.117773	训练正确率: 95.28%	校验正确率: 95.60%
    (tensor(4785), 5000)
    训练周期: 6 [12800/60000 (21%)]	Loss: 0.111244	训练正确率: 95.45%	校验正确率: 95.70%
    (tensor(4799), 5000)
    训练周期: 6 [19200/60000 (32%)]	Loss: 0.146247	训练正确率: 95.37%	校验正确率: 95.98%
    (tensor(4790), 5000)
    训练周期: 6 [25600/60000 (43%)]	Loss: 0.178848	训练正确率: 95.53%	校验正确率: 95.80%
    (tensor(4800), 5000)
    训练周期: 6 [32000/60000 (53%)]	Loss: 0.134815	训练正确率: 95.56%	校验正确率: 96.00%
    (tensor(4797), 5000)
    训练周期: 6 [38400/60000 (64%)]	Loss: 0.134774	训练正确率: 95.59%	校验正确率: 95.94%
    (tensor(4794), 5000)
    训练周期: 6 [44800/60000 (75%)]	Loss: 0.217827	训练正确率: 95.60%	校验正确率: 95.88%
    (tensor(4802), 5000)
    训练周期: 6 [51200/60000 (85%)]	Loss: 0.090015	训练正确率: 95.57%	校验正确率: 96.04%
    (tensor(4788), 5000)
    训练周期: 6 [57600/60000 (96%)]	Loss: 0.167978	训练正确率: 95.56%	校验正确率: 95.76%
    (tensor(4800), 5000)
    训练周期: 7 [0/60000 (0%)]	Loss: 0.059424	训练正确率: 98.44%	校验正确率: 96.00%
    (tensor(4805), 5000)
    训练周期: 7 [6400/60000 (11%)]	Loss: 0.071551	训练正确率: 95.90%	校验正确率: 96.10%
    (tensor(4801), 5000)
    训练周期: 7 [12800/60000 (21%)]	Loss: 0.082574	训练正确率: 95.86%	校验正确率: 96.02%
    (tensor(4803), 5000)
    训练周期: 7 [19200/60000 (32%)]	Loss: 0.135515	训练正确率: 95.80%	校验正确率: 96.06%
    (tensor(4803), 5000)
    训练周期: 7 [25600/60000 (43%)]	Loss: 0.162218	训练正确率: 95.81%	校验正确率: 96.06%
    (tensor(4799), 5000)
    训练周期: 7 [32000/60000 (53%)]	Loss: 0.173746	训练正确率: 95.81%	校验正确率: 95.98%
    (tensor(4801), 5000)
    训练周期: 7 [38400/60000 (64%)]	Loss: 0.080197	训练正确率: 95.88%	校验正确率: 96.02%
    (tensor(4818), 5000)
    训练周期: 7 [44800/60000 (75%)]	Loss: 0.196690	训练正确率: 95.82%	校验正确率: 96.36%
    (tensor(4823), 5000)
    训练周期: 7 [51200/60000 (85%)]	Loss: 0.168024	训练正确率: 95.80%	校验正确率: 96.46%
    (tensor(4810), 5000)
    训练周期: 7 [57600/60000 (96%)]	Loss: 0.210090	训练正确率: 95.86%	校验正确率: 96.20%
    (tensor(4807), 5000)
    训练周期: 8 [0/60000 (0%)]	Loss: 0.189429	训练正确率: 89.06%	校验正确率: 96.14%
    (tensor(4815), 5000)
    训练周期: 8 [6400/60000 (11%)]	Loss: 0.104325	训练正确率: 96.13%	校验正确率: 96.30%
    (tensor(4806), 5000)
    训练周期: 8 [12800/60000 (21%)]	Loss: 0.038240	训练正确率: 96.10%	校验正确率: 96.12%
    (tensor(4807), 5000)
    训练周期: 8 [19200/60000 (32%)]	Loss: 0.067944	训练正确率: 96.41%	校验正确率: 96.14%
    (tensor(4816), 5000)
    训练周期: 8 [25600/60000 (43%)]	Loss: 0.111516	训练正确率: 96.34%	校验正确率: 96.32%
    (tensor(4825), 5000)
    训练周期: 8 [32000/60000 (53%)]	Loss: 0.116992	训练正确率: 96.32%	校验正确率: 96.50%
    (tensor(4814), 5000)
    训练周期: 8 [38400/60000 (64%)]	Loss: 0.179410	训练正确率: 96.33%	校验正确率: 96.28%
    (tensor(4820), 5000)
    训练周期: 8 [44800/60000 (75%)]	Loss: 0.144883	训练正确率: 96.34%	校验正确率: 96.40%
    (tensor(4812), 5000)
    训练周期: 8 [51200/60000 (85%)]	Loss: 0.033405	训练正确率: 96.32%	校验正确率: 96.24%
    (tensor(4821), 5000)
    训练周期: 8 [57600/60000 (96%)]	Loss: 0.067156	训练正确率: 96.31%	校验正确率: 96.42%


    (tensor(4821), 5000)
    训练周期: 9 [0/60000 (0%)]	Loss: 0.031377	训练正确率: 100.00%	校验正确率: 96.42%
    (tensor(4831), 5000)
    训练周期: 9 [6400/60000 (11%)]	Loss: 0.017983	训练正确率: 96.43%	校验正确率: 96.62%
    (tensor(4845), 5000)
    训练周期: 9 [12800/60000 (21%)]	Loss: 0.148903	训练正确率: 96.58%	校验正确率: 96.90%
    (tensor(4834), 5000)
    训练周期: 9 [19200/60000 (32%)]	Loss: 0.030355	训练正确率: 96.61%	校验正确率: 96.68%
    (tensor(4822), 5000)
    训练周期: 9 [25600/60000 (43%)]	Loss: 0.228225	训练正确率: 96.57%	校验正确率: 96.44%
    (tensor(4830), 5000)
    训练周期: 9 [32000/60000 (53%)]	Loss: 0.193084	训练正确率: 96.46%	校验正确率: 96.60%
    (tensor(4839), 5000)
    训练周期: 9 [38400/60000 (64%)]	Loss: 0.075586	训练正确率: 96.49%	校验正确率: 96.78%
    (tensor(4828), 5000)
    训练周期: 9 [44800/60000 (75%)]	Loss: 0.119831	训练正确率: 96.53%	校验正确率: 96.56%
    (tensor(4834), 5000)
    训练周期: 9 [51200/60000 (85%)]	Loss: 0.208166	训练正确率: 96.57%	校验正确率: 96.68%
    (tensor(4833), 5000)
    训练周期: 9 [57600/60000 (96%)]	Loss: 0.147563	训练正确率: 96.58%	校验正确率: 96.66%
    (tensor(4838), 5000)
    训练周期: 10 [0/60000 (0%)]	Loss: 0.101206	训练正确率: 95.31%	校验正确率: 96.76%
    (tensor(4837), 5000)
    训练周期: 10 [6400/60000 (11%)]	Loss: 0.236028	训练正确率: 97.00%	校验正确率: 96.74%
    (tensor(4834), 5000)
    训练周期: 10 [12800/60000 (21%)]	Loss: 0.040362	训练正确率: 96.84%	校验正确率: 96.68%
    (tensor(4853), 5000)
    训练周期: 10 [19200/60000 (32%)]	Loss: 0.123055	训练正确率: 96.89%	校验正确率: 97.06%
    (tensor(4845), 5000)
    训练周期: 10 [25600/60000 (43%)]	Loss: 0.073337	训练正确率: 96.95%	校验正确率: 96.90%
    (tensor(4842), 5000)
    训练周期: 10 [32000/60000 (53%)]	Loss: 0.108402	训练正确率: 96.87%	校验正确率: 96.84%
    (tensor(4836), 5000)
    训练周期: 10 [38400/60000 (64%)]	Loss: 0.102884	训练正确率: 96.85%	校验正确率: 96.72%
    (tensor(4850), 5000)
    训练周期: 10 [44800/60000 (75%)]	Loss: 0.040080	训练正确率: 96.84%	校验正确率: 97.00%
    (tensor(4854), 5000)
    训练周期: 10 [51200/60000 (85%)]	Loss: 0.082359	训练正确率: 96.81%	校验正确率: 97.08%
    (tensor(4837), 5000)
    训练周期: 10 [57600/60000 (96%)]	Loss: 0.137477	训练正确率: 96.81%	校验正确率: 96.74%
    (tensor(4853), 5000)
    训练周期: 11 [0/60000 (0%)]	Loss: 0.079699	训练正确率: 96.88%	校验正确率: 97.06%
    (tensor(4838), 5000)
    训练周期: 11 [6400/60000 (11%)]	Loss: 0.199870	训练正确率: 96.58%	校验正确率: 96.76%
    (tensor(4834), 5000)
    训练周期: 11 [12800/60000 (21%)]	Loss: 0.141394	训练正确率: 96.79%	校验正确率: 96.68%
    (tensor(4839), 5000)
    训练周期: 11 [19200/60000 (32%)]	Loss: 0.005506	训练正确率: 96.83%	校验正确率: 96.78%
    (tensor(4856), 5000)
    训练周期: 11 [25600/60000 (43%)]	Loss: 0.081504	训练正确率: 96.92%	校验正确率: 97.12%
    (tensor(4849), 5000)
    训练周期: 11 [32000/60000 (53%)]	Loss: 0.064143	训练正确率: 96.91%	校验正确率: 96.98%
    (tensor(4847), 5000)
    训练周期: 11 [38400/60000 (64%)]	Loss: 0.045931	训练正确率: 96.98%	校验正确率: 96.94%
    (tensor(4854), 5000)
    训练周期: 11 [44800/60000 (75%)]	Loss: 0.046780	训练正确率: 96.96%	校验正确率: 97.08%
    (tensor(4850), 5000)
    训练周期: 11 [51200/60000 (85%)]	Loss: 0.035275	训练正确率: 97.00%	校验正确率: 97.00%
    (tensor(4859), 5000)
    训练周期: 11 [57600/60000 (96%)]	Loss: 0.170923	训练正确率: 96.97%	校验正确率: 97.18%
    (tensor(4860), 5000)
    训练周期: 12 [0/60000 (0%)]	Loss: 0.154155	训练正确率: 95.31%	校验正确率: 97.20%
    (tensor(4852), 5000)
    训练周期: 12 [6400/60000 (11%)]	Loss: 0.219587	训练正确率: 97.00%	校验正确率: 97.04%
    (tensor(4851), 5000)
    训练周期: 12 [12800/60000 (21%)]	Loss: 0.106742	训练正确率: 96.96%	校验正确率: 97.02%
    (tensor(4863), 5000)
    训练周期: 12 [19200/60000 (32%)]	Loss: 0.202796	训练正确率: 97.01%	校验正确率: 97.26%
    (tensor(4864), 5000)
    训练周期: 12 [25600/60000 (43%)]	Loss: 0.120019	训练正确率: 97.08%	校验正确率: 97.28%
    (tensor(4871), 5000)
    训练周期: 12 [32000/60000 (53%)]	Loss: 0.052749	训练正确率: 97.11%	校验正确率: 97.42%
    (tensor(4850), 5000)
    训练周期: 12 [38400/60000 (64%)]	Loss: 0.100737	训练正确率: 97.13%	校验正确率: 97.00%
    (tensor(4841), 5000)
    训练周期: 12 [44800/60000 (75%)]	Loss: 0.122833	训练正确率: 97.09%	校验正确率: 96.82%
    (tensor(4853), 5000)
    训练周期: 12 [51200/60000 (85%)]	Loss: 0.225710	训练正确率: 97.08%	校验正确率: 97.06%
    (tensor(4859), 5000)
    训练周期: 12 [57600/60000 (96%)]	Loss: 0.098465	训练正确率: 97.06%	校验正确率: 97.18%
    (tensor(4857), 5000)
    训练周期: 13 [0/60000 (0%)]	Loss: 0.034278	训练正确率: 100.00%	校验正确率: 97.14%
    (tensor(4854), 5000)
    训练周期: 13 [6400/60000 (11%)]	Loss: 0.075550	训练正确率: 97.37%	校验正确率: 97.08%
    (tensor(4846), 5000)
    训练周期: 13 [12800/60000 (21%)]	Loss: 0.127165	训练正确率: 97.33%	校验正确率: 96.92%
    (tensor(4863), 5000)
    训练周期: 13 [19200/60000 (32%)]	Loss: 0.074782	训练正确率: 97.27%	校验正确率: 97.26%
    (tensor(4855), 5000)
    训练周期: 13 [25600/60000 (43%)]	Loss: 0.070970	训练正确率: 97.32%	校验正确率: 97.10%
    (tensor(4868), 5000)
    训练周期: 13 [32000/60000 (53%)]	Loss: 0.025788	训练正确率: 97.30%	校验正确率: 97.36%
    (tensor(4868), 5000)
    训练周期: 13 [38400/60000 (64%)]	Loss: 0.077453	训练正确率: 97.30%	校验正确率: 97.36%
    (tensor(4868), 5000)
    训练周期: 13 [44800/60000 (75%)]	Loss: 0.094940	训练正确率: 97.30%	校验正确率: 97.36%
    (tensor(4866), 5000)
    训练周期: 13 [51200/60000 (85%)]	Loss: 0.104054	训练正确率: 97.31%	校验正确率: 97.32%
    (tensor(4867), 5000)
    训练周期: 13 [57600/60000 (96%)]	Loss: 0.183157	训练正确率: 97.29%	校验正确率: 97.34%
    (tensor(4858), 5000)
    训练周期: 14 [0/60000 (0%)]	Loss: 0.067517	训练正确率: 96.88%	校验正确率: 97.16%
    (tensor(4877), 5000)
    训练周期: 14 [6400/60000 (11%)]	Loss: 0.030041	训练正确率: 97.45%	校验正确率: 97.54%
    (tensor(4848), 5000)
    训练周期: 14 [12800/60000 (21%)]	Loss: 0.033137	训练正确率: 97.33%	校验正确率: 96.96%
    (tensor(4861), 5000)
    训练周期: 14 [19200/60000 (32%)]	Loss: 0.073892	训练正确率: 97.27%	校验正确率: 97.22%
    (tensor(4883), 5000)
    训练周期: 14 [25600/60000 (43%)]	Loss: 0.143710	训练正确率: 97.25%	校验正确率: 97.66%
    (tensor(4870), 5000)
    训练周期: 14 [32000/60000 (53%)]	Loss: 0.102851	训练正确率: 97.30%	校验正确率: 97.40%
    (tensor(4874), 5000)
    训练周期: 14 [38400/60000 (64%)]	Loss: 0.103107	训练正确率: 97.32%	校验正确率: 97.48%
    (tensor(4871), 5000)
    训练周期: 14 [44800/60000 (75%)]	Loss: 0.046912	训练正确率: 97.35%	校验正确率: 97.42%
    (tensor(4873), 5000)
    训练周期: 14 [51200/60000 (85%)]	Loss: 0.031629	训练正确率: 97.35%	校验正确率: 97.46%
    (tensor(4840), 5000)
    训练周期: 14 [57600/60000 (96%)]	Loss: 0.071023	训练正确率: 97.38%	校验正确率: 96.80%
    (tensor(4863), 5000)
    训练周期: 15 [0/60000 (0%)]	Loss: 0.082912	训练正确率: 96.88%	校验正确率: 97.26%
    (tensor(4857), 5000)
    训练周期: 15 [6400/60000 (11%)]	Loss: 0.151896	训练正确率: 97.59%	校验正确率: 97.14%
    (tensor(4872), 5000)
    训练周期: 15 [12800/60000 (21%)]	Loss: 0.003805	训练正确率: 97.59%	校验正确率: 97.44%
    (tensor(4869), 5000)
    训练周期: 15 [19200/60000 (32%)]	Loss: 0.021541	训练正确率: 97.46%	校验正确率: 97.38%
    (tensor(4873), 5000)
    训练周期: 15 [25600/60000 (43%)]	Loss: 0.113653	训练正确率: 97.44%	校验正确率: 97.46%
    (tensor(4877), 5000)
    训练周期: 15 [32000/60000 (53%)]	Loss: 0.042688	训练正确率: 97.49%	校验正确率: 97.54%
    (tensor(4873), 5000)
    训练周期: 15 [38400/60000 (64%)]	Loss: 0.079477	训练正确率: 97.46%	校验正确率: 97.46%
    (tensor(4875), 5000)
    训练周期: 15 [44800/60000 (75%)]	Loss: 0.090744	训练正确率: 97.44%	校验正确率: 97.50%
    (tensor(4877), 5000)
    训练周期: 15 [51200/60000 (85%)]	Loss: 0.062950	训练正确率: 97.44%	校验正确率: 97.54%
    (tensor(4877), 5000)
    训练周期: 15 [57600/60000 (96%)]	Loss: 0.049283	训练正确率: 97.44%	校验正确率: 97.54%
    (tensor(4874), 5000)
    训练周期: 16 [0/60000 (0%)]	Loss: 0.046523	训练正确率: 98.44%	校验正确率: 97.48%
    (tensor(4885), 5000)
    训练周期: 16 [6400/60000 (11%)]	Loss: 0.026660	训练正确率: 97.99%	校验正确率: 97.70%
    (tensor(4865), 5000)
    训练周期: 16 [12800/60000 (21%)]	Loss: 0.046088	训练正确率: 97.64%	校验正确率: 97.30%
    (tensor(4869), 5000)
    训练周期: 16 [19200/60000 (32%)]	Loss: 0.074679	训练正确率: 97.71%	校验正确率: 97.38%
    (tensor(4869), 5000)
    训练周期: 16 [25600/60000 (43%)]	Loss: 0.088331	训练正确率: 97.72%	校验正确率: 97.38%
    (tensor(4871), 5000)
    训练周期: 16 [32000/60000 (53%)]	Loss: 0.065556	训练正确率: 97.78%	校验正确率: 97.42%
    (tensor(4855), 5000)
    训练周期: 16 [38400/60000 (64%)]	Loss: 0.060337	训练正确率: 97.78%	校验正确率: 97.10%
    (tensor(4881), 5000)
    训练周期: 16 [44800/60000 (75%)]	Loss: 0.062322	训练正确率: 97.72%	校验正确率: 97.62%
    (tensor(4873), 5000)
    训练周期: 16 [51200/60000 (85%)]	Loss: 0.012551	训练正确率: 97.71%	校验正确率: 97.46%
    (tensor(4881), 5000)
    训练周期: 16 [57600/60000 (96%)]	Loss: 0.003745	训练正确率: 97.74%	校验正确率: 97.62%
    (tensor(4879), 5000)
    训练周期: 17 [0/60000 (0%)]	Loss: 0.056673	训练正确率: 98.44%	校验正确率: 97.58%
    (tensor(4878), 5000)
    训练周期: 17 [6400/60000 (11%)]	Loss: 0.032328	训练正确率: 97.62%	校验正确率: 97.56%
    (tensor(4867), 5000)
    训练周期: 17 [12800/60000 (21%)]	Loss: 0.045340	训练正确率: 97.55%	校验正确率: 97.34%
    (tensor(4876), 5000)
    训练周期: 17 [19200/60000 (32%)]	Loss: 0.086995	训练正确率: 97.60%	校验正确率: 97.52%
    (tensor(4880), 5000)
    训练周期: 17 [25600/60000 (43%)]	Loss: 0.187141	训练正确率: 97.62%	校验正确率: 97.60%
    (tensor(4880), 5000)
    训练周期: 17 [32000/60000 (53%)]	Loss: 0.030895	训练正确率: 97.60%	校验正确率: 97.60%
    (tensor(4886), 5000)
    训练周期: 17 [38400/60000 (64%)]	Loss: 0.165245	训练正确率: 97.62%	校验正确率: 97.72%
    (tensor(4865), 5000)
    训练周期: 17 [44800/60000 (75%)]	Loss: 0.134593	训练正确率: 97.62%	校验正确率: 97.30%
    (tensor(4878), 5000)
    训练周期: 17 [51200/60000 (85%)]	Loss: 0.029219	训练正确率: 97.63%	校验正确率: 97.56%


    (tensor(4884), 5000)
    训练周期: 17 [57600/60000 (96%)]	Loss: 0.100540	训练正确率: 97.62%	校验正确率: 97.68%
    (tensor(4880), 5000)
    训练周期: 18 [0/60000 (0%)]	Loss: 0.038466	训练正确率: 100.00%	校验正确率: 97.60%
    (tensor(4889), 5000)
    训练周期: 18 [6400/60000 (11%)]	Loss: 0.020327	训练正确率: 97.90%	校验正确率: 97.78%
    (tensor(4882), 5000)
    训练周期: 18 [12800/60000 (21%)]	Loss: 0.104299	训练正确率: 97.91%	校验正确率: 97.64%
    (tensor(4874), 5000)
    训练周期: 18 [19200/60000 (32%)]	Loss: 0.030101	训练正确率: 97.93%	校验正确率: 97.48%
    (tensor(4886), 5000)
    训练周期: 18 [25600/60000 (43%)]	Loss: 0.020961	训练正确率: 97.88%	校验正确率: 97.72%
    (tensor(4879), 5000)
    训练周期: 18 [32000/60000 (53%)]	Loss: 0.057023	训练正确率: 97.80%	校验正确率: 97.58%
    (tensor(4883), 5000)
    训练周期: 18 [38400/60000 (64%)]	Loss: 0.016640	训练正确率: 97.82%	校验正确率: 97.66%
    (tensor(4887), 5000)
    训练周期: 18 [44800/60000 (75%)]	Loss: 0.033335	训练正确率: 97.78%	校验正确率: 97.74%
    (tensor(4872), 5000)
    训练周期: 18 [51200/60000 (85%)]	Loss: 0.035585	训练正确率: 97.74%	校验正确率: 97.44%
    (tensor(4892), 5000)
    训练周期: 18 [57600/60000 (96%)]	Loss: 0.035861	训练正确率: 97.76%	校验正确率: 97.84%
    (tensor(4890), 5000)
    训练周期: 19 [0/60000 (0%)]	Loss: 0.042498	训练正确率: 100.00%	校验正确率: 97.80%
    (tensor(4878), 5000)
    训练周期: 19 [6400/60000 (11%)]	Loss: 0.054951	训练正确率: 98.21%	校验正确率: 97.56%
    (tensor(4885), 5000)
    训练周期: 19 [12800/60000 (21%)]	Loss: 0.044866	训练正确率: 97.96%	校验正确率: 97.70%
    (tensor(4886), 5000)
    训练周期: 19 [19200/60000 (32%)]	Loss: 0.245563	训练正确率: 98.00%	校验正确率: 97.72%
    (tensor(4886), 5000)
    训练周期: 19 [25600/60000 (43%)]	Loss: 0.052003	训练正确率: 98.00%	校验正确率: 97.72%
    (tensor(4883), 5000)
    训练周期: 19 [32000/60000 (53%)]	Loss: 0.061380	训练正确率: 97.96%	校验正确率: 97.66%
    (tensor(4882), 5000)
    训练周期: 19 [38400/60000 (64%)]	Loss: 0.153816	训练正确率: 97.95%	校验正确率: 97.64%
    (tensor(4884), 5000)
    训练周期: 19 [44800/60000 (75%)]	Loss: 0.038608	训练正确率: 97.95%	校验正确率: 97.68%
    (tensor(4876), 5000)
    训练周期: 19 [51200/60000 (85%)]	Loss: 0.207824	训练正确率: 97.94%	校验正确率: 97.52%
    (tensor(4896), 5000)
    训练周期: 19 [57600/60000 (96%)]	Loss: 0.051280	训练正确率: 97.91%	校验正确率: 97.92%



```python
#绘制训练过程的误差曲线，校验集和测试集上的错误率。
plt.figure(figsize = (10, 7))
plt.plot(record) #record记载了每一个打印周期记录的训练和校验数据集上的准确度
plt.xlabel('Steps')
plt.ylabel('Error rate')
```




    Text(0, 0.5, 'Error rate')




    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_14_1.png)
    


### 2.3 在测试集上进行分类


```python
#在测试集上分批运行，并计算总的正确率
net.eval() #标志模型当前为运行阶段
vals = [] #记录准确率所用列表

#对测试数据集进行循环
for data, target in test_loader:
    data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
    output = net(data) #将特征数据喂入网络，得到分类的输出
    val = rightness(output, target) #获得正确样本数以及总样本数
    vals.append(val) #记录结果

#计算准确率
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 100. * rights[0].numpy() / rights[1]
print(right_rate)
```

    99.14



```python
#随便从测试集中读入一张图片，并检验模型的分类结果，并绘制出来
idx = 4089
muteimg = test_dataset[idx][0].numpy()
plt.imshow(muteimg[0,...])
print('标签是：',test_dataset[idx][1])

```

    标签是： 7



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_17_1.png)
    


## 三、解剖卷积神经网络

我们可以将训练好的卷积神经网络net进行解剖。我们主要关注一下几个问题：
1. 第一层卷积核训练得到了什么；
2. 第一层卷积核是如何在训练的过程中随着时间的演化而发生变化
3. 在输入特定图像的时候，第一层卷积核所对应的4个featuremap是什么样子
4. 第二层卷积核都是什么东西？
5. 对于给定输入图像，第二层卷积核所对应的那些featuremaps都是什么样？

本文件是集智AI学园http://campus.swarma.org 出品的“火炬上的深度学习”第III课的配套源代码

### 3.1 第一层卷积核、演化与特征图

#### 第一层卷积核


```python
#提取第一层卷积层的卷积核
plt.figure(figsize = (10, 7))
for i in range(4):
    plt.subplot(1,4,i + 1)
    plt.axis('off')
    plt.imshow(net.conv1.weight.data.numpy()[i,0,...]) #提取第一层卷积核中的权重值，注意conv1是net的属性
```


    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_22_0.png)
    



```python
plt.subplot(1,4,2)
```




    <AxesSubplot:>




    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_23_1.png)
    


#### 滤波器的演化


```python
# 将记录在容器中的卷积核权重历史演化数据打印出来
i = 0
for tup in weights:
    if i % 10 == 0 :
        layer1 = tup[0]
        fig = plt.figure(figsize = (10, 7))
        for j in range(4):
            plt.subplot(1, 4, j + 1)
            plt.axis('off')
            plt.imshow(layer1.numpy()[j,0,...])
    i += 1

```


    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_0.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_1.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_2.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_3.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_4.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_5.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_6.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_7.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_8.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_9.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_10.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_11.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_12.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_13.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_14.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_15.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_16.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_17.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_18.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_25_19.png)
    


#### 绘制第一层特征图


```python
#调用net的retrieve_features方法可以抽取出喂入当前数据后吐出来的所有特征图（第一个卷积和第二个卷积层）

#首先定义读入的图片

#它是从test_dataset中提取第idx个批次的第0个图，其次unsqueeze的作用是在最前面添加一维，
#目的是为了让这个input_x的tensor是四维的，这样才能输入给net。补充的那一维表示batch。
input_x = test_dataset[idx][0].unsqueeze(0) 
feature_maps = net.retrieve_features(input_x) #feature_maps是有两个元素的列表，分别表示第一层和第二层卷积的所有特征图

plt.figure(figsize = (10, 7))

#有四个特征图，循环把它们打印出来
for i in range(4):
    plt.subplot(1,4,i + 1)
    plt.axis('off')
    plt.imshow(feature_maps[0][0, i,...].data.numpy())
    
```


    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_27_0.png)
    


### 3.2 绘制第二层卷积核与特征图


```python
# 绘制第二层的卷积核，每一列对应一个卷积核，一共8个卷积核
plt.figure(figsize = (15, 10))
for i in range(4):
    for j in range(8):
        plt.subplot(4, 8, i * 8 + j + 1)
        plt.axis('off')
        plt.imshow(net.conv2.weight.data.numpy()[j, i,...])
        
```


    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_29_0.png)
    



```python
# 绘制第二层的特征图，一共八个
plt.figure(figsize = (10, 7))
for i in range(8):
    plt.subplot(2,4,i + 1)
    plt.axis('off')
    plt.imshow(feature_maps[1][0, i,...].data.numpy())
    
```


    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_30_0.png)
    


### 3.3 卷积神经网络的鲁棒性试验

该试验设计如下：我们随机挑选一张测试图像，把它往左平移w个单位，然后：
1. 考察分类结果是否变化
2. 考察两层卷积对应的featuremap们有何变化


```python
# 提取中test_dataset中的第idx个批次的第0个图的第0个通道对应的图像，定义为a。
a = test_dataset[idx][0][0]

# 平移后的新图像将放到b中。根据a给b赋值。
b = torch.zeros(a.size()) #全0的28*28的矩阵
w = 3 #平移的长度为3个像素

# 对于b中的任意像素i,j，它等于a中的i,j+w这个位置的像素
for i in range(a.size()[0]):
    for j in range(0, a.size()[1] - w):
        b[i, j] = a[i, j + w]

# 将b画出来
muteimg = b.numpy()
plt.axis('off')
plt.imshow(muteimg)

# 把b喂给神经网络，得到分类结果pred（prediction是预测的每一个类别的概率的对数值），并把结果打印出来
prediction = net(b.unsqueeze(0).unsqueeze(0))
pred = torch.max(prediction.data, 1)[1]
print(pred)

#提取b对应的featuremap结果
feature_maps = net.retrieve_features(b.unsqueeze(0).unsqueeze(0))

plt.figure(figsize = (10, 7))
for i in range(4):
    plt.subplot(1,4,i + 1)
    plt.axis('off')
    plt.imshow(feature_maps[0][0, i,...].data.numpy())

plt.figure(figsize = (10, 7))
for i in range(8):
    plt.subplot(2,4,i + 1)
    plt.axis('off')

    plt.imshow(feature_maps[1][0, i,...].data.numpy())
```

    tensor([0])



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_32_1.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_32_2.png)
    



    
![png](%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_files/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E5%99%A8_minst_convnet_32_3.png)
    



```python
def extract_data(filename, num_images):
    # filename: 文件存放的路径，num_images: 读入的图片个数
    """将图像解压缩展开，读入成一个4维的张量： [image index（图像的编码）, y（纵坐标）, x（横坐标）, channels（通道）].
    我们将数组中的数值范围从原来的[0, 255]降低到了[-0.5, 0.5]范围内
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        return data

def extract_labels(filename, num_images):
    """将label的数据文件解压缩，并将label读成64位的整数"""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

# 将数据解压缩并存储到数组中，60000张图片，60000个label，测试集中有10000张图片
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

# 将一部分图片（VALIDATION_SIZE=5000张）定为校验数据集
validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:]
num_epochs = NUM_EPOCHS
```


```python
#随便从数据集中读入一张图片，并绘制出来
idx = 100
muteimg = train_data[idx, 0, :, :]
plt.imshow(muteimg)
```

## 附录：不用dataloader版本的卷积神经网络

如果不喜欢用PyTorch自带的dataset，dataloader等加载数据，可以用下面的代码来完成卷积神经网络的计算
本文件是集智AI学园http://campus.swarma.org 出品的“火炬上的深度学习”第III课的配套源代码


```python
import gzip
import os
import sys

from six.moves import urllib

#定义一系列常数
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/' #图像数据如果没下载，可以从这个地址下载
WORK_DIRECTORY = 'data' #存储的路径名
IMAGE_SIZE = 28 #每张图片的大小尺寸
NUM_CHANNELS = 1  #每张图片的通道数
PIXEL_DEPTH = 255 #像素的深度0-255
NUM_LABELS = 10 #手写数字，一共十种
VALIDATION_SIZE = 5000 #校验数据集大小
NUM_EPOCHS = 20 # 训练的循环周期
BATCH_SIZE = 64 #batch的大小
EVAL_FREQUENCY = 100 #每隔100个batch进行一次校验集计算

%matplotlib inline
```


```python
#下载图像文件，如果文件已经存在，那么就不下载。
def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.isdir(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        size = os.path.getsize(filepath)
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath
# Get the data.
train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
```


```python
def extract_data(filename, num_images):
    # filename: 文件存放的路径，num_images: 读入的图片个数
    """将图像解压缩展开，读入成一个4维的张量： [image index（图像的编码）, y（纵坐标）, x（横坐标）, channels（通道）].
    我们将数组中的数值范围从原来的[0, 255]降低到了[-0.5, 0.5]范围内
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        return data

def extract_labels(filename, num_images):
    """将label的数据文件解压缩，并将label读成64位的整数"""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

# 将数据解压缩并存储到数组中，60000张图片，60000个label，测试集中有10000张图片
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

# 将一部分图片（VALIDATION_SIZE=5000张）定为校验数据集
validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:]
num_epochs = NUM_EPOCHS
```

本文件是集智学园http://campus.swarma.org 出品的“火炬上的深度学习”第III课的配套源代码


```python
# 定义关键函数
# 我们在训练中用了一个Tensorflow图，在评价的过程中，我们拷贝了这个图
def error_rate(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，labels是数据之中的正确答案"""
    predictions = np.argmax(predictions, 1)
    return 100.0 - (
      100.0 *
      np.sum( predictions == labels) /
      predictions.shape[0])

def eval_in_batches(data, net):
    """计算得到预测精度，运行对所有的数据集中的小撮数据进行."""
    size = data.shape[0]
    if size < BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    
    #一个小撮一个小撮地进行预测的计算
    for begin in range(0, size, BATCH_SIZE):
        end = begin + BATCH_SIZE
        if end <= size:
            inputs = data[begin:end]
            inputs = torch.from_numpy(inputs)
            inputs = inputs.clone().detach().requires_grad_(True)
            outputs = net(inputs)
            predictions[begin:end, :] = outputs.data.numpy()
        else:
            inputs = data[-BATCH_SIZE:]
            inputs = torch.from_numpy(inputs)
            inputs = inputs.clone().detach().requires_grad_(True)
            outputs = net(inputs)
            
            batch_predictions = outputs.data.numpy()
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions
```


```python
error_rate_rec = []

#获得训练集尺寸
train_size = train_labels.shape[0]

print('Initialized!')

net = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 开始训练循环，一共进行int(num_epochs * train_size) // BATCH_SIZE次
print(int(num_epochs * train_size) // BATCH_SIZE)
for step in range(int(num_epochs * train_size) // BATCH_SIZE):
    # 计算当前应该访问训练集中的哪一部分数据
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    batch_data = torch.from_numpy(batch_data)
    batch_labels = torch.from_numpy(batch_labels)
    
    inputs, labels = batch_data.clone().detach(), batch_labels.clone().detach()
    
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    #print(loss)
    loss.backward()
    optimizer.step()
    
    if step % EVAL_FREQUENCY == 0:
        #每间隔EVAL_FREQUENCY就打印一次预测结果
        
        print('第 %d (epoch %.2f) 步' %
              (step, float(step) * BATCH_SIZE / train_size))
        #print('损失函数值: %.3f, 学习率: %.6f' % (l, lr))
        predictions = outputs.data.numpy()
        #print(predictions.shape)
        err_train = error_rate(predictions, labels.data.numpy())
        prediction = eval_in_batches(validation_data, net)
        err_valid = error_rate(prediction, validation_labels)
        print('训练测试率: %.1f%%' % err_train)
        print('校验集测试率: %.1f%%' % err_valid)
        error_rate_rec.append([err_train,err_valid])
        sys.stdout.flush()
# 在测试集上实验
prediction = eval_in_batches(test_data, net)
test_error = error_rate(prediction, test_labels)
print('测试误差: %.1f%%' % test_error)

```


```python
plt.plot(error_rate_rec)
plt.xlabel('Batches')
plt.ylabel('Error Rate')
```


```python

#将测试集喂进去，得到计算结果，看一看预测的准确度
right = 0
batch_num = len(test_data) // BATCH_SIZE
for i in range(batch_num):
    inputs = test_data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    inputs = torch.from_numpy(inputs)
    inputs = inputs.clone().detach()
    result = net(inputs)
    result = result.data.numpy()
    
    labels = test_labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    right += np.sum(np.argmax(result, 1) == labels)
print(right / float(batch_num * BATCH_SIZE))
```

本文件是集智学园http://campus.swarma.org 出品的“火炬上的深度学习”第III课的配套源代码


```python

```
