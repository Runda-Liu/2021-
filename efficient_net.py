# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
import torch
import random
import PIL
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import argparse
import warnings
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式
from efficientnet_pytorch import EfficientNet
from trash_dataloader import TrashDataset
from label_smooth import LabelSmoothSoftmaxCE
import os
from trash_dataloader import TrashDataset
from torch.utils.data import DataLoader
# from data_Augmentation import *
from ev_toolkit import plot_tool
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(1)
warnings.filterwarnings("ignore")
# Number of classes in the dataset
num_classes = 146

# Batch size for training (change depending on how much memory you have)
batch_size = 128 # 批处理尺寸(batch_size)

# Number of epochs to train for
EPOCH = 30

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
# feature_extract = True
feature_extract = False
# 超参数设置
pre_epoch = 0  # 定义已经遍历数据集的次数
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]
)


# 获取数据集文件夹下的类别文件夹名称赋值为type_list
train_dir = './split_data/train'
valid_dir = './split_data/val'
# transform = get_transforms(input_size=224, test_size=224, backbone=None)
# 构建MyDataset实例

base_dir = "../../../../home/data"
num = os.listdir(base_dir)
temp = os.path.join(base_dir,num[0])
# 仅用于编码测试
type_lst = os.listdir(temp)
train_data = TrashDataset(data_dir = train_dir,transform=train_transforms,classes = type_lst)
valid_data = TrashDataset(data_dir = valid_dir,transform=val_transforms,classes = type_lst)
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,pin_memory=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False,pin_memory=True)
# 用Adam优化器
try:
    pre_model_path = 'project/train/pre-trained-models/efficient_bowenqiange/models.pkl'
    net = torch.load(pre_model_path)
    print('load pretrained model')
except:
    net = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = net._fc.in_features
net._fc = nn.Linear(num_ftrs, num_classes)
# 显示网络信息
print(net)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练使用多GPU，测试单GPU
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
net.to(device)
# 读取网络信息
# Send the model to GPU
net = net.to(device)

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch DeepNetwork Training')
args = parser.parse_args()
params_to_update = net.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

def save_plot(train_curve,train_acc,tp):
    train_x = list(range(len(train_curve)))
    train_loss = np.array(train_curve)
    train_acc = np.array(train_acc)
    train_iters = len(train_loader)
    fig_loss = plt.figure(figsize = (10,6))
    plt.plot(train_x, train_loss)
    plot_tool.update_plot(name='loss', img=plt.gcf())
    fig_loss.savefig('../result-graphs/{}loss.png'.format(tp))

    fig_acc = plt.figure(figsize = (10,6))
    plt.plot(train_x, train_acc)
    plot_tool.update_plot(name='acc', img=plt.gcf())
    fig_acc = plt.gcf()
    fig_acc.savefig('../result-graphs/{}acc.png'.format(tp))
    print('acc-loss曲线绘制已完成')            
            

def main():
    train_curve = list()
    valid_curve = list()
    train_acc = list()
    valid_acc = list()
    ii = 0
    LR = 0.001  # 学习率
    best_acc = 0  # 初始化best test accuracy
    print("Start Training, DeepNetwork!")  # 定义遍历数据集的次数

    # criterion
#     criterion = LabelSmoothSoftmaxCE()
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1,momentum=0.9,
                                weight_decay=1e-4)

    for epoch in range(pre_epoch, EPOCH):
        # scheduler.step(epoch)
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader):
            # 准备数据
            length = len(train_loader)
            input, target = data
            input, target = input.to(device), target.to(device)
            # 训练
            optimizer.zero_grad()
            # forward + backward
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            train_curve.append(loss.item())
            train_acc.append(correct / total)
            if i%10 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                         100. * float(correct) / float(total)))

        with torch.no_grad():
            correct = 0
            total = 0
            for data in valid_loader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).cpu().sum()
            print('test_acc：%.3f%%' % (100. * float(correct) / float(total)))
            acc = 100. * float(correct) / float(total)
            print(acc)
            valid_curve.append(loss.item())
            valid_acc.append(correct / total)
            path_model = '../models/efficient_NMD_final/models.pkl'
            print('{} is save！'.format(path_model))
            torch.save(net, path_model)
            
    print("Training Finished, TotalEPOCH=%d" % EPOCH)
    return train_curve,train_acc,valid_curve,valid_acc





if __name__ == "__main__":
    train_curve,train_acc,valid_curve,valid_acc = main()
    save_plot(train_curve,train_acc,tp='train')
    save_plot(valid_curve,valid_acc,tp='val')


