"""
@Author: suqiulin
@Email: 72405483@cityu-dg.edu.cn
@Date: 2024/12/3
"""
import os
from datetime import datetime
import torch
from networks.ssp import ssp
from utils.loss import bceLoss
from utils.options import get_options
from utils.tdataloader import get_val_dict_infos, get_train_loader_by_names
from utils.util import set_random_seed, poly_lr, print_per_pic_res

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(train_loader, model, optimizer, current_epoch, train_opt):
    model.train()
    global step
    epoch_step = 0
    #100 pics, 10 pics/batch,  total_batch_size = 10
    total_batch_size = len(train_loader)
    loss_all = 0
    device = torch.device("mps")  # 指定Apple GPU设备
    save_path = train_opt.save_path
    total_epoch = train_opt.epochs
    try:
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.to(device)  # 将图像数据移动到Apple GPU设备
            preds = model(images).ravel()
            labels = labels.to(device)  # 将标签数据移动到Apple GPU设备
            loss1 = bceLoss()
            loss = loss1(preds, labels)
            loss.backward()
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if batch_index % 200 == 0 or batch_index == total_batch_size or batch_index == 1:
                print(f'{datetime.now()} Epoch [{current_epoch:03d}/{total_epoch:03d}], Batch [{batch_index:04d}/{total_batch_size:04d}], Total_loss: {loss.data:.4f}')
        loss_all /= epoch_step
        if current_epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + f'macos_{current_epoch}th_epoch.pth')

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')

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
        #res > 0.5 is nature pic, res < 0.5 is ai pic
        current_prediction = (((res > 0.5) & (labels == 1)) | ((res < 0.5) & (labels == 0))).cpu().numpy()
        # 一个batch是64,数据大的情况下，每隔print_gap*64个图片才打印一次识别
        if (print_gap is not None) and batch_index % print_gap == 0:
            print_per_pic_res(batch_index, len(images), current_prediction, label_name, label_loader)
        right_label_image += current_prediction.sum()
    print(f'{label_name} accu:{right_label_image / label_size}')
    return right_label_image


def varify_for_train(val_loader, model, epoch, save_path, print_gap=10):
    model.eval()
    global best_epoch, best_accu
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
            print(f'val on:{name}, Epoch:{epoch}, Accuracy:{accu}')
    total_accu = total_right_image / total_image
    if epoch == 1:
        best_accu = total_accu
        best_epoch = 1
        torch.save(model.state_dict(), save_path + 'macos_first_epoch_best.pth')
        print(f'Save state_dict successfully! Best epoch:{epoch}.')
    else:
        if total_accu > best_accu:
            best_accu = total_accu
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + f'macos_net_epoch_best_{int(best_accu)}.pth')
            print(f'Save state_dict successfully! Best epoch:{epoch}.')
    print(
        f'Epoch:{epoch},Accuracy:{total_accu}, bestEpoch:{best_epoch}, bestAccu:{best_accu}')

def get_train_options(image_root):
    #train 和 val的父文件夹
    train_opt = get_options()
    train_opt.image_root = image_root
    train_opt.load = None
    train_opt.epochs = 10
    return train_opt


def get_val_options(image_root):
    val_opt = get_options()
    val_opt.image_root = image_root
    # val_opt.isTrain = False
    # val_opt.isVal = True
    # blur
    val_opt.blur_prob = 0
    val_opt.blur_sig = [1]
    # jpg
    val_opt.jpg_prob = 0
    val_opt.jpg_method = ['pil']
    val_opt.jpg_qual = [90]
    return val_opt


if __name__ == '__main__':
    set_random_seed()
    image_root = '/Users/sequel/linkcodes/homework/ml/data/genImage'
    train_opt = get_train_options(image_root)
    val_opt = get_val_options(image_root)
    print('load data...')
    train_loader = get_train_loader_by_names(train_opt)
    #batch_size, if total=100, each batch contains 10 pics, batch_size =10，
    val_loader = get_val_dict_infos(val_opt)

    # 加载模型并移动到Apple GPU设备
    model = ssp()
    device = torch.device("mps")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), train_opt.lr)
    save_path = train_opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    step = 0
    best_epoch = 0
    best_accu = 0
    print("Start train")
    for epoch in range(1, train_opt.epochs + 1):
        cur_lr = poly_lr(optimizer, train_opt.lr, epoch, train_opt.epochs)
        train(train_loader, model, optimizer, epoch, train_opt)
        varify_for_train(val_loader, model, epoch, save_path)
