# 猫狗识别器
## 环境和数据准备
### resnet-50
这篇文章带大家使用PyTorch、SwanLab、Gradio三个开源工具，完成从数据集准备、代码编写、可视化训练到构建Demo网页的全过程。   
原网页:https://www.zhihu.com/question/478789646/answer/3362166804
### 创建文件目录  
打开1个文件夹，新建下面这5个文件：  
![alt 文件夹](https://pica.zhimg.com/80/v2-5c36fbe5d05d7beb8d05e8217d56ab93_1440w.webp?source=2c26e567)  
它们各自的作用分别是：  
- checkpoint：这个文件夹用于存储训练过程中生成的模型权重。
- datasets：这个文件夹用于放置数据集。
- app.py：运行Gradio Demo的Python脚本。
- load_datasets.py：负责载入数据集，包含了数据的预处理、加载等步骤，确保数据以适当的格式提供给模型使用。
- train.py：模型训练的核心脚本。它包含了模型的载入、训练循环、损失函数的选择、优化器的配置等关键组成部分，用于指导如何使用数据来训练模型。  
### 下载猫狗分类数据
集数据集来源是Modelscope上的猫狗分类数据集，包含275张图像的数据集和70张图像的测试集，一共不到10MB。 百度网盘：链接: https://pan.baidu.com/s/1qYa13SxFM0AirzDyFMy0mQ 提取码: 1ybm  
将数据集放入datasets文件夹：  
![alt 解压数据集](https://picx.zhimg.com/80/v2-f9c447442d4b3ab583fe35a76edef04f_1440w.webp?source=2c26e567)  
## 训练部分
### 2.1 load_datasets.py
我们首先需要创建1个类DatasetLoader，它的作用是完成数据集的读取和预处理，我们将它写在load_datasets.py中。 在写这个类之前，先分析一下数据集。  
在datasets目录下，train.csv和val.csv分别记录了训练集和测试集的图像相对路径（第一列是图像的相对路径，第二列是标签，0代表猫，1代表狗）：  
![alt 数据集分析](https://picx.zhimg.com/80/v2-7e57ecb881ccb5e52dece6d3424a6c93_1440w.webp?source=2c26e567)  
那么我们的目标就很明确：
1. 解析这两个csv文件，获取图像相对路径和标签
2. 根据相对路径读取图像
3. 对图像做预处理
3. 返回预处理后的图像和对应标签
明确了目标后，现在我们开始写DatasetLoader类：
```
import csv
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(self, csv_path):
        self.csv_file = csv_path
        with open(self.csv_file, 'r') as file:
            self.data = list(csv.reader(file))

        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def preprocess_image(self, image_path):
        full_path = os.path.join(self.current_dir, 'datasets', image_path)
        image = Image.open(full_path)
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return image_transform(image)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = self.preprocess_image(image_path)
        return image, int(label)

    def __len__(self):
        return len(self.data)
```
DatasetLoader类由四个部分组成：  
1.`__init__`：包含1个输入参数csv_path，在外部传入csv_path后，将读取后的数据存入self.data中。self.current_dir则是获取了当前代码所在目录的绝对路径，为后续读取图像做准备。
2. preprocess_image：此函数用于图像预处理。首先，它构造图像文件的绝对路径，然后使用PIL库打开图像。接着，定义了一系列图像变换：调整图像大小至256x256、转换图像为张量、对图像进行标准化处理，最终，返回预处理后的图像。
3. `__getitem__`：当数据集类被循环调用时，__getitem__方法会返回指定索引index的数据，即图像和标签。首先，它根据索引从self.data中取出图像路径和标签。然后，调用prepogress_image方法来处理图像数据。最后，将处理后的图像数据和标签转换为整型后返回。
4. `__len__`：用于返回数据集的总图像数量。
### 2.2 载入数据集
从本节开始，代码将写在train.py中  
```
from torch.utils.data import DataLoader
from load_datasets import DatasetLoader

batch_size = 8

TrainDataset = DatasetLoader("datasets/train.csv")
ValDataset = DatasetLoader("datasets/val.csv")
TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)
```  
我们传入那两个csv文件的路径实例化DatasetLoader类，然后用PyTorch的DataLoader做一层封装。 DataLoader可以再传入两个参数：   
**batch_size**：定义了每个数据批次包含多少张图像。在深度学习中，我们通常不会一次性地处理所有数据，而是将数据划分为小批次。这有助于模型更快地学习，并且还可以节省内存。在这里我们定义batch_size = 8，即每个批次将包含8个图像。   
**shuffle**：定义了是否在每个循环轮次（epoch）开始时随机打乱数据。这通常用于训练数据集以保证每个epoch的数据顺序不同，从而帮助模型更好地泛化。如果设置为True，那么在每个epoch开始时，数据将被打乱。在这里我们让训练时打乱，测试时不打乱。
### 2.3 载入ResNet50模型
模型我们选用经典的ResNet50，模型的具体原理本文不细说，重点放在工程实现上。
我们使用torchvision来创建1个resnet50模型，并载入预训练权重：
```
from torchvision.models import ResNet50_Weights
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 加载预训练的ResNet50模型
```
因为猫狗分类是个2分类任务，而torchvision提供的resnet50默认是1000分类，所以我们需要把模型最后的全连接层的输出维度替换为2：  
```
from torchvision.models import ResNet50_Weights
num_classes=2
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)# 加载预训练的ResNet50模型
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)# 将全连接层的输出维度替换为num_classes
```  
### 2.4 设置cuda/mps/cpu
如果你的电脑是英伟达显卡，那么cuda可以极大加速你的训练；
如果你的电脑是Macbook Apple Sillicon（M系列芯片），那么mps同样可以极大加速你的训练；
如果都不是，那就选用cpu：  
```
#检测是否支持mps
try:
    use_mps = torch.backends.mps.is_available()
except AttributeError:
    use_mps = False

#检测是否支持cuda
if torch.cuda.is_available():
    device = "cuda"
elif use_mps:
    device = "mps"
else:
    device = "cpu"
```
将模型加载到对应的device中：
```
model.to(torch.device(device))
```
### 2.5 设置超参数、优化器、损失函数
**超参数**
设置训练轮次为20轮，学习率为1e-4，训练批次为8，分类数为2分类。
```
num_epochs = 20
lr = 1e-4
batch_size = 8
num_classes = 2
```
**损失函数与优化器**
设置损失函数为交叉熵损失，优化器为Adam。
```
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```
### 2.6 初始化SwanLab
**设置初始化配置参数**

swanlab库使用swanlab.init设置实验名、实验介绍、记录超参数以及日志文件的保存位置。
打开可视化看板需要根据日志文件完成。
```
import swanlab

swanlab.init(
    # 设置实验名
    experiment_name="ResNet50",
    # 设置实验介绍
    description="Train ResNet50 for cat and dog classification.",
    # 记录超参数
    config={
        "model": "resnet50",
        "optim": "Adam",
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_class": num_classes,
        "device": device,
    },
    # 设置日志文件的保存位置
    logdir='./logs'
)
```
### 2.7 训练函数
我们定义1个训练函数train： 
```
作者：林泽毅
链接：https://www.zhihu.com/question/478789646/answer/3362166804
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

def train(model, device, train_dataloader, optimizer, criterion, epoch):
    model.train()
    for iter, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(TrainDataLoader),
                                                                      loss.item()))
        swanlab.log({"train_loss": loss.item()})
```
训练的逻辑很简单：我们循环调用train_dataloader，每次取出1个batch_size的图像和标签，传入到resnet50模型中得到预测结果，将结果和标签传入损失函数中计算交叉熵损失，最后根据损失计算反向传播，Adam优化器执行模型参数更新，循环往复。 在训练中我们最关心的指标是损失值loss，所以我们用swanlab.log跟踪它的变化。
### 2.8 测试函数
我们定义1个测试函数test：
```
作者：林泽毅
链接：https://www.zhihu.com/question/478789646/answer/3362166804
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

def test(model, device, test_dataloader, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print('Accuracy: {:.2f}%'.format(accuracy))
    swanlab.log({"test_acc": accuracy})
```
测试的逻辑同样很简单：我们循环调用test_dataloader，将测试集的图像传入到resnet50模型中得到预测结果，与标签进行对比，计算整体的准确率。 在测试中我们最关心的指标是准确率accuracy，所以我们用swanlab.log跟踪它的变化。至此，我们完成已完成了绝大多数的代码，现在是时候将它们组合起来，开始训练！
### 2.9 完整训练代码
我们一共训练num_epochs轮，每4轮进行测试，并在最后保存权重文件：
```
for epoch in range(1, num_epochs + 1):
    train(model, device, TrainDataLoader, optimizer, criterion, epoch)
    if epoch % 4 == 0: 
        accuracy = test(model, device, ValDataLoader, epoch)

if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")
torch.save(model.state_dict(), 'checkpoint/latest_checkpoint.pth')
print("Training complete")
```
### 2.10 开始训练！

我们运行train.py：  
![alt 开始训练](https://pic1.zhimg.com/80/v2-685ecfb796f3157adebaeaf0513e2a16_1440w.webp?source=2c26e567)
在训练开始时候，你的训练目录下应该会多1个logs文件夹，里面存放着你的训练过程数据。打开终端，输入swanlab watch --logdir logs开启SwanLab实验看板：
![alt 打开swanlab面板](https://picx.zhimg.com/80/v2-1c375928d958edd56b35e79199e698ca_1440w.webp?source=2c26e567)
点开 http:127.0.0.1:5092 ，将在浏览器中看到实验看板。
默认页面是Project DashBoard，包含了项目信息和一个对比实验表格：
![alt 面板界面](https://picx.zhimg.com/80/v2-3fef30631d1c9826d8dfb36b2ca2cf4b_1440w.webp?source=2c26e567)
至此我们完成了模型的训练和测试，得到了1个表现非常棒的猫狗分类模型，权重保存在了checkpoint目录下。
接下来，我们就基于训练好的权重，创建1个Demo网页吧～  
### 3. Gradio演示程序
Gradio是一个开源的Python库，旨在帮助数据科学家、研究人员和从事机器学习领域的开发人员快速创建和共享用于机器学习模型的用户界面。
在这里我们使用Gradio来构建一个猫狗分类的Demo界面，编写app.py程序：  
```
作者：林泽毅
链接：https://www.zhihu.com/question/478789646/answer/3362166804
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

import gradio as gr
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision


# 加载与训练中使用的相同结构的模型
def load_model(checkpoint_path, num_classes):
    # 加载预训练的ResNet50模型
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    model = torchvision.models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model


# 加载图像并执行必要的转换的函数
def process_image(image, image_size):
    # Define the same transforms as used during training
    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocessing(image).unsqueeze(0)
    return image


# 预测图像类别并返回概率的函数
def predict(image):
    classes = {'0': 'cat', '1': 'dog'}  # Update or extend this dictionary based on your actual classes
    image = process_image(image, 256)  # Using the image size from training
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1).squeeze()  # Apply softmax to get probabilities
    # Mapping class labels to probabilities
    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    return class_probabilities


# 定义到您的模型权重的路径
checkpoint_path = 'checkpoint/latest_checkpoint.pth'
num_classes = 2
model = load_model(checkpoint_path, num_classes)

# 定义Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Cat vs Dog Classifier",
)

if __name__ == "__main__":
    iface.launch()
```
运行程序后，会出现以下输出：
![alt](https://picx.zhimg.com/80/v2-052ef603c8978d0012d4c8a03388bf8d_1440w.webp?source=2c26e567)
点开链接，出现猫狗分类的Demo网页：
![app界面](https://pica.zhimg.com/80/v2-43525ab2e6f1db3115a3d035990ec73d_1440w.webp?source=2c26e567)
用猫和狗的图片试试：
![猫](https://pica.zhimg.com/80/v2-7a0338e55844c5ee78f70b2baedd453c_1440w.webp?source=2c26e567)
![狗](https://picx.zhimg.com/80/v2-b5c290d8d78279d76fa87a3c1e233be4_1440w.webp?source=2c26e567)
效果很完美！
至此，我们完成了用PyTorch、SwanLab、Gradio三个开源工具训练1个猫狗分类模型的全部过程。