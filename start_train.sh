#!/bin/bash
project_root_dir=/project/train/src_repo

cd ${project_root_dir}
# echo '环境安装已完成！'
# echo '正在准备训练！'
mkdir -p /project/train/models/efficient_NMD_final
python3.6 -u ${project_root_dir}/data_analysis.py
echo '类别分布绘图已完成！'
python3.6 -u ${project_root_dir}/preprocess.py
# python3.6 -u ${project_root_dir}/trash_resnet.py
python3.6 -u ${project_root_dir}/efficient_net.py
# echo 'resnet网络训练完成！'
# echo 'efficient网络训练开始！'
