"""
@Author: suqiulin
@Email: 72405483@cityu-dg.edu.cn
@Date: 2024/12/5
"""
import os
import torch

from networks.ssp import ssp
from utils.options import get_options
from utils.tdataloader import get_val_dict_infos
from utils.util import set_random_seed, print_per_pic_res

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def traversal_label_loader(label_loader, label_size, print_gap=None):
    #/xxxxx/imagenet_cityu_test/val/ai/yy.jpg
    demo_path = label_loader.dataset.img_list[0]
    #保留'ai'
    label_name = os.path.basename(os.path.dirname(demo_path))
    right_label_image = 0
    for batch_index, (images, labels) in enumerate(label_loader):
        if torch.backends.mps.is_available():
            images = images.to(torch.device("mps"))
            labels = labels.to(torch.device("mps"))
        res = model(images)
        res = torch.sigmoid(res).ravel()
        current_prediction = (((res > 0.5) & (labels == 1)) | ((res < 0.5) & (labels == 0))).cpu().numpy()
        # 一个batch是64,数据大的情况下，每隔print_gap*64个图片才打印一次识别
        if (print_gap is not None) and batch_index % print_gap == 0 :
            print_per_pic_res(batch_index, len(images), current_prediction, label_name, label_loader)
        right_label_image += current_prediction.sum()
    print(f'{label_name} accu:{right_label_image / label_size}')
    return right_label_image

def val(val_loader, model, print_gap=None):
    model.eval()
    total_right_image = total_image = 0
    with torch.no_grad():
        for loader in val_loader:
            name, val_ai_loader, ai_size, val_nature_loader, nature_size = loader['name'], loader['val_ai_loader'], loader['ai_size'], loader['val_nature_loader'], loader['nature_size']
            print("val on:", name)
            right_ai_image = traversal_label_loader(val_ai_loader, ai_size, print_gap)
            right_nature_image = traversal_label_loader(val_nature_loader, nature_size, print_gap)
            accu = (right_ai_image + right_nature_image) / (ai_size + nature_size)
            total_right_image += right_ai_image + right_nature_image
            total_image += ai_size + nature_size
            print(f'val on:{name}, Accuracy:{accu}')
    total_accu = total_right_image / total_image
    print(f'total accuracy:{total_accu}')

# %% [markdown]
# # 1. Only need to modify is just the below cell, other cells should remain still.
# # 2. The code will choose GPU or CPU automatically

# %%
def rewrite_test_opt():
    test_dataset_path = '/Users/sequel/linkcodes/homework/ml/data/genImage'
    test_opt = get_options()
    test_opt.image_root = test_dataset_path
    test_opt.dataset_names = ['my_test']
    #务必保证snapshot在同级目录下，如果加载不成功要修改为绝对路径
    #------------------------------------------------------------!!!2.可能需要修改的点2-已训练模型的路径(大概率不用)
    test_opt.load = '../snapshot/sortnet/Net_epoch_best.pth'
    #每隔64*print_gap个图片打印改批次识别情况
    test_opt.print_gap = 20
    return test_opt


if __name__ == '__main__':
    set_random_seed()
    # ------------------------------------------------------------!!!1.可能需要修改的点-测试数据集所在的文件夹【绝对路径】
    test_opt = rewrite_test_opt()

    # load data
    print('load data...')
    val_loader = get_val_dict_infos(test_opt)

    devicelink = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = ssp()
    model.to(devicelink)

    if test_opt.load is None:
        print("not found model")
    model.load_state_dict(torch.load(test_opt.load, map_location=devicelink))
    print('load model from', test_opt.load)
    print("Start test")

    val(val_loader, model, test_opt.print_gap)

