"""
@Author: suqiulin
@Email: 72405483@cityu-dg.edu.cn
@Date: 2024/12/3
"""
import os
import torch
from tabulate import tabulate

from utils.util import set_random_seed, poly_lr
from utils.tdataloader import get_single_test_loader, get_val_loader
from options import TrainOptions
from networks.ssp import ssp
import pandas as pd
from utils.loss import bceLoss
from datetime import datetime
import numpy as np

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.isVal = True
    # blur
    val_opt.blur_prob = 0
    val_opt.blur_sig = [1]
    # jpg
    val_opt.jpg_prob = 0
    val_opt.jpg_method = ['pil']
    val_opt.jpg_qual = [90]

    return val_opt

def print_per_pic_res(index, size, current_prediction, label_name, val_label_loader):
    print(f"current checking {label_name} datas:")
    data = []
    for i in range(len(current_prediction)):
        img_path = val_label_loader.dataset.img_list[index * size + i]
        last_path = os.path.basename(img_path)
        is_label = current_prediction[i]
        data.append([last_path, is_label])
    df = pd.DataFrame(data, columns=["img_name", f"is_{label_name}"])
    print(tabulate(df, headers='keys', tablefmt='grid'))

def traversal_label_loader(label_loader, label_size):
    #/xxxxx/imagenet_cityu_test/val/ai/yy.jpg
    demo_path = label_loader.dataset.img_list[0]
    #保留'ai'
    label_name = os.path.basename(os.path.dirname(demo_path))
    right_label_image = 0
    for index, (images, labels) in enumerate(label_loader):
        res = model(images)
        res = torch.sigmoid(res).ravel()
        current_prediction = (((res > 0.5) & (labels == 1)) | ((res < 0.5) & (labels == 0))).cpu().numpy()
        print_per_pic_res(index, len(images), current_prediction, label_name, label_loader)
        right_label_image += current_prediction.sum()
    print(f'{label_name} accu:{right_label_image / label_size}')
    return right_label_image

def val(val_loader, model):
    model.eval()
    total_right_image = total_image = 0
    with torch.no_grad():
        for loader in val_loader:
            name, val_ai_loader, ai_size, val_nature_loader, nature_size = loader['name'], loader['val_ai_loader'], loader['ai_size'], loader['val_nature_loader'], loader['nature_size']
            print("val on:", name)
            right_ai_image = traversal_label_loader(val_ai_loader, ai_size)
            right_nature_image = traversal_label_loader(val_nature_loader, nature_size)
            accu = (right_ai_image + right_nature_image) / (ai_size + nature_size)
            total_right_image += right_ai_image + right_nature_image
            total_image += ai_size + nature_size
            print(f'val on:{name}, Accuracy:{accu}')
    total_accu = total_right_image / total_image
    print(f'total accuracy:{total_accu}')


if __name__ == '__main__':
    set_random_seed()
    # train and val options
    opt = TrainOptions().parse()
    val_opt = get_val_opt()

    # load data
    print('load data...')
    val_loader = get_single_test_loader(val_opt)

    # 不再进行GPU相关的设备设置
    # 直接实例化模型，不需要转移到GPU上（即去掉.cuda()方法调用）
    model = ssp()
    if opt.load is not None:
        # 加载模型参数时，使用map_location指定加载到CPU上
        model.load_state_dict(torch.load(opt.load, map_location=torch.device('cpu')))
        print('load model from', opt.load)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Start test")
    val(val_loader, model)
