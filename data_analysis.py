import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from ev_toolkit import plot_tool

train_dir = '../../../../home/data'
num_classes = os.listdir(train_dir)
train_dir = os.path.join(train_dir,num_classes[0])
type_lst = os.listdir(train_dir)
num = []
for i in type_lst:
    classes = '{}/{}/*.jpg'.format(train_dir, i)
    path_file_number=glob.glob(classes)#或者指定文件下个数
    num.append(len(path_file_number))

fig = plt.figure(figsize = (24,8))
plt.bar(np.arange(len(num)), np.array(num),width=0.4)
plt.xticks(np.arange(len(num)),type_lst,rotation = 90)
plt.ylabel('number of pictures')
plt.title('Distribution of type')
plot_tool.update_plot(name='loss', img=plt.gcf())
fig.savefig('../result-graphs/type_distribution.png',dpi = 150,bbox_inches = 'tight')
plt.show()
