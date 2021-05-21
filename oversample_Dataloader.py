import os
import random
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from data_Augmentation import *
import numpy as np
import glob

class TrashDataset(Dataset):
    def __init__(self,transform,data_dir,classes,data_ratio=1.0,ret_name = False):
        self.ret_name = ret_name
        self.transform = transform
        self.cls_to_ind_dict = dict()
        self.ind_to_cls_dict = list()
        self.cls_list = list()a
        self.img_list = list()
        self.cls_num = dict()
        self.max_num = list()
        
        num = []
        for i in classes:
            im_pth = '{}/{}/*.jpg'.format(data_dir, i)
            path_file_number=glob.glob(im_pth)#或者指定文件下个数
            num.append(len(path_file_number))
        num = np.array(num)
        self.max_num.append(np.max(num))

        for idx, cls in enumerate(classes):
            img_list_temp, cls_list_temp = [], []
            self.cls_to_ind_dict[cls] = idx
            self.ind_to_cls_dict.append(cls)
            if cls == 'normal':
                img_list = sorted(os.listdir(os.path.join(data_dir, cls)))
                self.cls_num[cls] = len(img_list)
                
                for img_fp in img_list:
                    self.img_list.append(os.path.join(data_dir, cls,img_fp))
                    self.cls_list.append(idx)
                print(cls, '=====================')
                print(len(img_list))

            else:
                img_list = sorted(os.listdir(os.path.join(data_dir,cls)))
                self.cls_num[cls] = len(img_list)
                for img_fp in img_list:
                    img_list_temp.append(os.path.join(data_dir,cls,img_fp))
                    cls_list_temp.append(idx)
                #扩充n倍
                n = self.max_num[0] / self.cls_num[cls]
                img_list_temp = [val for val in img_list_temp for i in range(int(n))]
                cls_list_temp = [val for val in cls_list_temp for i in range(int(n))]
                print(cls, len(cls_list_temp))
                self.cls_num[cls] = len(img_list_temp)

                print(cls, '======================')
                print(len(img_list_temp))

            self.img_list = img_list_temp + self.img_list
            self.cls_list = cls_list_temp + self.cls_list

    def __getitem__(self, index):
        name = self.img_list[index]
        img = Image.open(name)
#         num = random.randint(0,6)
#         img = self.trans_list[num](img)
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        label = self.cls_list[index]
        if self.ret_name:
            return img, label, name
        else:
            return img, label

    def __len__(self):
        return len(self.img_list)
