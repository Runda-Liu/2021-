import os
import time
import torch.nn as nn
import torch
import numpy as np
import random
import glob
from skimage import io,transform,color
from sklearn.utils import shuffle
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from trash_dataloader import TrashDataset
# from oversample_Dataloader import TrashDataset
from torchvision import datasets, models, transforms
from focal_loss import FocalLoss
import time
import seaborn
import sys
import glob
import threading
import time
import matplotlib.pyplot as plt
from data_Augmentation import *
from ev_toolkit import plot_tool
from args import args
from label_smooth import LabelSmoothSoftmaxCE
from self_optimizer import get_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))
#设置随机种子
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(15)

# 参数设置
num_classes = 146
MAX_EPOCH = 25
BATCH_SIZE = 128
LR = 0.001
log_interval = 10
val_interval = 1
start_epoch = -1
lr_decay_step = 7

base_dir = "../../../../home/data"
num = os.listdir(base_dir)
temp = os.path.join(base_dir,num[0])
# 仅用于编码测试
type_lst = os.listdir(temp)

# step 1/5 数据

# 获取数据集文件夹下的类别文件夹名称赋值为type_list
train_dir = './split_data/train'
valid_dir = './split_data/val'

transform = get_transforms(input_size=224, test_size=224, backbone=None)

# 构建MyDataset实例

train_data = TrashDataset(data_dir = train_dir,transform=transform['train'],classes = type_lst)
valid_data = TrashDataset(data_dir = valid_dir,transform=transform['val'],classes = type_lst)
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False)
# step 2/5 构建模型

try:
    pre_model_path = '/project/train/pre-trained-models/models.pkl'
    # if os.path.exists(pre_model_path):
    net = torch.load(pre_model_path)
    print('load pretrained model')
except:
    net = models.resnet50(pretrained=True)

# path_pretrained_model = ("/project/train/pre-trained-models/models.pkl")
# path_pretrained_model = ("models/resnet50.pth")
# state_dict_load = torch.load(path_pretrained_model)
# net.load_state_dict(state_dict_load)
#替换fc层s
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_classes)
net.cuda()
#加载参数



#step 3/5 损失函数
# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothSoftmaxCE()
# criterion = FocalLoss()
#step 4/5 优化器
optimizer = get_optimizer(net, args)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=False)

# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)


#step 5/5 训练
train_curve = list()
valid_curve = list()
train_acc = list()
valid_acc = list()
for epoch in range(start_epoch + 1, MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()


    for i, data in enumerate(train_loader):
        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        train_acc.append(correct / total)
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

            # if flag_m1:

#             print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(epoch, net.conv1.weight[0, 0, ...]))

      # 更新学习率
    
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                loss_val += loss.item()

            loss_val_mean = loss_val/len(valid_loader)
            #如果测试效果好就保存
            if len(valid_acc) != 0:
                if correct_val / total_val > np.max(np.array(valid_acc)) and loss_val_mean < np.min(np.array(valid_curve)):
                    t = time.strftime('%Y.%m.%d',time.localtime(time.time()))
                    path_model = '../models/resnetfinalbowen/models.pkl'
                    print('{} is save！'.format(path_model))
                    torch.save(net, path_model)
        
            valid_curve.append(loss_val_mean)
            valid_acc.append(correct_val / total_val)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))

        net.train()
    scheduler.step(loss_val_mean)
print('train finish!')


#将绘制的曲线图进行可视化
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
save_plot(train_curve,train_acc,tp='train')
save_plot(valid_curve,valid_acc,tp='val')



