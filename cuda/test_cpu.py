"""
@Author: suqiulin
@Email: 72405483@cityu-dg.edu.cn
@Date: 2024/12/3
"""
import os
import sys

import pandas as pd
import torch
from tabulate import tabulate

from networks.ssp import ssp
from utils.options import get_options
from utils.tdataloader import get_test_data_list
from utils.util import set_random_seed

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        print("good, current res is", res)
        current_prediction = (((res > 0.5) & (labels == 1)) | ((res < 0.5) & (labels == 0))).cpu().numpy()
        print_per_pic_res(index, len(images), current_prediction, label_name, label_loader)
        right_label_image += current_prediction.sum()
    print(f'{label_name} accu:{right_label_image / label_size}')
    return right_label_image

def val(test_data_list, model):
    model.eval()
    total_right_image = total_image = 0
    with torch.no_grad():
        for dict_loader in test_data_list:
            name, val_ai_loader, ai_size, val_nature_loader, nature_size = dict_loader['name'], dict_loader['val_ai_loader'], dict_loader['ai_size'], dict_loader['val_nature_loader'], dict_loader['nature_size']
            print("val on:", name)
            right_ai_image = traversal_label_loader(val_ai_loader, ai_size)
            right_nature_image = traversal_label_loader(val_nature_loader, nature_size)
            accu = (right_ai_image + right_nature_image) / (ai_size + nature_size)
            total_right_image += right_ai_image + right_nature_image
            total_image += ai_size + nature_size
            print(f'val on:{name}, Accuracy:{accu}')
    total_accu = total_right_image / total_image
    print(f'total accuracy:{total_accu}')

def rewrite_test_opt(test_dataset_path):
    test_opt = get_options()
    test_opt.choices = [2]
    test_opt.isTrain = False
    test_opt.isVal = True
    # blur
    test_opt.blur_prob = 0
    test_opt.blur_sig = [1]
    # jpg
    test_opt.jpg_prob = 0
    test_opt.jpg_method = ['pil']
    test_opt.jpg_qual = [90]
    test_opt.image_root = os.path.dirname(test_dataset_path)
    test_opt.test_set_dir = test_dataset_path
    #务必保证snapshot在同级目录下，如果加载不成功要修改为绝对路径
    #------------------------------------------------------------!!!2.可能需要修改的点2-已训练模型的路径(大概率不用)
    test_opt.load =  './cityu.pth'
    #每隔64*print_gap个图片打印改批次识别情况
    test_opt.print_gap = 20
    return test_opt

if __name__ == '__main__':
    set_random_seed()
    # ------------------------------------------------------------!!!1.可能需要修改的点-测试数据集所在的文件夹【绝对路径】
    # All you need to do is just to modify this variate
    # test_dataset_absolute_path = ''
    test_dataset_absolute_path = '/Users/sequel/linkcodes/homework/ml/data/genImage/imagenet_cityu_test'

    if not test_dataset_absolute_path:
        print("Not specify the absolute path of test dataset")
        sys.exit(1)
    test_opt = rewrite_test_opt(test_dataset_absolute_path)

    if torch.cuda.is_available() and test_opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')

    # load data
    print('load data...')
    val_loader = get_test_data_list(test_opt)

    enable_cuda = torch.cuda.is_available()
    model = ssp().cuda() if enable_cuda else ssp()
    device = torch.device('cuda' if enable_cuda else 'cpu')

    if test_opt.load is None:
        print("not found model")
    model.load_state_dict(torch.load(test_opt.load, map_location=device))
    print('load model from', test_opt.load)
    print("Start test")

    val(val_loader, model, test_opt.print_gap)
