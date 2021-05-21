import os
import random
import numpy as np
import glob
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
#确定是在测试还是在训练

trash_name = {}
f = open('/project/train/src_repo/class.txt','r')
a = f.read()
trash_name = eval(a)
f.close()


class TrashDataset(Dataset):
    def __init__(self, data_dir, transform, classes):
        self.data_info = self.get_img_info(classes,data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform
        self.labelname = trash_name
        self.classes = classes
        self.data_dir = data_dir
        


    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(classes,data_dir):
        num = []
        for i in classes:
            im_pth = '{}/{}/*.jpg'.format(data_dir, i)
            path_file_number=glob.glob(im_pth)#或者指定文件下个数
            num.append(len(path_file_number))
        num = np.array(num)
        max_num = np.max(num)
        
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                

                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                if len(img_names) < max_num:
                    n = max_num / len(img_names)
                    img_names = [val for val in img_names for i in range(int(n))]

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = trash_name[sub_dir]
                    data_info.append((path_img, int(label)))
        if len(data_info) == 0:
            raise Exception('\n data dir: {} is empty!'.format(data_dir))

        return data_info