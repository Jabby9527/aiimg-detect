import os

import pandas as pd
import torch
import numpy as np
import random
from tabulate import tabulate

def poly_lr(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        cur_lr = lr
    return cur_lr


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def print_per_pic_res(batch_index, size, current_prediction, label_name, val_label_loader):
    print(f"checking on {batch_index}th batch {label_name} datas:")
    data = []
    for i in range(len(current_prediction)):
        img_path = val_label_loader.dataset.img_list[batch_index * size + i]
        last_path = os.path.basename(img_path)
        is_label = current_prediction[i]
        data.append([last_path, is_label])
    df = pd.DataFrame(data, columns=["img_name", f"is_{label_name}"])
    print(tabulate(df, headers='keys', tablefmt='grid'))
    # print(df)