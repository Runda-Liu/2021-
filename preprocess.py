# 工具类
import os
import random
import shutil
from shutil import copy2
import numpy as np
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
set_seed(15)

def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.2):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val三个文件夹进行
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_folder)
                # print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))


if __name__ == '__main__':
    base_dir = "../../../../home/data"
    num = os.listdir(base_dir)
    src_data_folder = os.path.join(base_dir,num[0])
    os.makedirs("./split_data/",exist_ok=True)
    target_data_folder = "./split_data/"
    data_set_split(src_data_folder, target_data_folder)

