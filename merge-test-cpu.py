# %% [markdown]
# # Course Project Group Infomation
# members:
# Qiulin Su - 723405483

# %%
# import depedencies
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from PIL import ImageFile
import random
import pandas as pd
from datetime import datetime
import os
import argparse
from tabulate import tabulate
import cv2
import random as rd
from random import random, choice
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO

# %%
#resnet.py
__all__ = ["ResNet", "resnet18", "resnet34",
           "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model


# %%
#srm_conv.py
class SRMConv2d_simple(nn.Module):

    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        return filters


class SRMConv2d_Separate(nn.Module):

    def __init__(self, inc, outc, learnable=False):
        super(SRMConv2d_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        # filters = np.repeat(filters, inc, axis=1)
        filters = np.repeat(filters, inc, axis=0)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        # print(filters.size())
        return filters


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    srm = SRMConv2d_simple()
    output = srm(x)
    output = np.array(output)
    print(output.shape)

# %%
#ssp.py
class ssp(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        self.srm = SRMConv2d_simple()
        self.disc = resnet50(pretrained=True)
        self.disc.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = F.interpolate(x, (256, 256), mode='bilinear')
        x = self.srm(x)
        x = self.disc(x)
        return x


if __name__ == '__main__':
    model = ssp(pretrain=True)
    print(model)


# %%
#utils.py
import random
def compute(patch):
    weight, height = patch.size
    m = weight
    res = 0
    patch = np.array(patch).astype(np.int64)
    diff_horizontal = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))
    diff_vertical = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))
    diff_diagonal = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))
    diff_diagonal += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))
    res = diff_horizontal + diff_vertical + diff_diagonal
    return res.sum()


def patch_img(img, patch_size, height):
    img_width, img_height = img.size
    num_patch = (height // patch_size) * (height // patch_size)
    patch_list = []
    min_len = min(img_height, img_width)
    rz = transforms.Resize((height, height))
    if min_len < patch_size:
        img = rz(img)
    rp = transforms.RandomCrop(patch_size)
    for i in range(num_patch):
        patch_list.append(rp(img))
    patch_list.sort(key=lambda x: compute(x), reverse=False)
    new_img = patch_list[0]
    return new_img

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


# %%
#dataloader.py
class Opt:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class genImageValDataset(Dataset):
    def __init__(self, image_root, image_dir, is_real, opt):
        super().__init__()
        self.opt = opt
        self.root = os.path.join(image_root, image_dir, "val")
        if is_real:
            self.img_path = os.path.join(self.root, 'nature')
            self.img_list = [os.path.join(self.img_path, f) for f in os.listdir(self.img_path)]
            self.img_len = len(self.img_list)
            self.labels = torch.ones(self.img_len)
        else:
            self.img_path = os.path.join(self.root, 'ai')
            self.img_list = [os.path.join(self.img_path, f) for f in os.listdir(self.img_path)]
            self.img_len = len(self.img_list)
            self.labels = torch.zeros(self.img_len)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        image = self.rgb_loader(self.img_list[index])
        label = self.labels[index]
        image = processing(image, self.opt)
        return image, label

    def __len__(self):
        return self.img_len

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def sample_randint(s):
    if len(s) == 1:
        return s[0]
    return rd.randint(s[0], s[1])


def gaussian_blur_gray(img, sigma):
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_randint(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def processing(img, opt):
    if opt.aug:
        aug = transforms.Lambda(
            lambda img: data_augment(img, opt)
        )
    else:
        aug = transforms.Lambda(
            lambda img: img
        )

    if opt.isPatch:
        patch_func = transforms.Lambda(
            lambda img: patch_img(img, opt.patch_size, opt.trainsize))
    else:
        patch_func = transforms.Resize((256, 256))

    trans = transforms.Compose([
        aug,
        patch_func,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return trans(img)


def get_single_loader(opt, image_dir, is_real): 
    val_dataset = genImageValDataset(opt.image_root, image_dir=image_dir, is_real=is_real, opt=opt)
    val_loader = DataLoader(val_dataset, batch_size=opt.val_batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader, len(val_dataset)


def get_test_data_list(opt):
    choices = opt.choices
    test_dir = opt.test_set_dir
    list_perdict_has_loader_info = []
    loader_info_dict = dict()
    if choices[0] == 2:
        print("val on:", test_dir)
        loader_info_dict['name'] = test_dir
        loader_info_dict['val_ai_loader'], loader_info_dict['ai_size'] = get_single_loader(opt, loader_info_dict['name'], is_real=False)
        loader_info_dict['val_nature_loader'], loader_info_dict['nature_size'] = get_single_loader(opt, loader_info_dict['name'], is_real=True)
        list_perdict_has_loader_info.append(loader_info_dict)
    return list_perdict_has_loader_info


# %%
"""
@Author: suqiulin
@Email: 72405483@cityu-dg.edu.cn
@Date: 2024/12/3
"""

"""Currently assumes jpg_prob, blur_prob 0 or 1"""

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
        images = images
        labels = labels
        res = model(images)
        res = torch.sigmoid(res).ravel()
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

def get_options():
    options = {
        'name': 'experiment_name',
        'rz_interp': 'bilinear',
        'blur_prob': 0,
        'blur_sig': [0, 1],
        'jpg_prob': 0,
        'jpg_method': ['pil', 'cv2'],
        'jpg_qual': [90, 100],
        'CropSize': 224,
        'batchsize': 64,
        'choices': [1],
        'epoch': 30,
        'lr': 1e-4,
        'trainsize': 256,
        'load': './snapshot/sortnet/cityu.pth',
        'image_root': '/root/autodl-fs/genImage',
        'save_path': './snapshot/sortnet/',
        'isPatch': False,
        'patch_size': 32,
        'aug': False,
        'gpu_id': '0',
        'log_name': 'log3.log',
        'val_interval': 1,
        'val_batchsize': 64,
        'test_set_dir': None
    }
    opt = Opt(**options)
    return opt

def rewrite_test_opt(test_dataset_path, load_model_path=None):
    test_opt = get_options()
    test_opt.choices = [2]
    test_opt.image_root = '/Users/sequel/linkcodes/homework/ml/data/genImage'
    test_opt.test_set_dir = test_dataset_path
    test_opt.load =  './cityu.pth' if load_model_path is None else load_model_path
    return test_opt

if __name__ == '__main__':
    set_random_seed()
    #修改为数据集所在的绝对路径，不是相对
    # input_test_dataset_path = input("输入测试集的文件夹名称(无绝对路径):")
    test_opt = rewrite_test_opt("imagenet_cityu_test")
    print(test_opt)

    # load data
    print('load data...')
    opt = test_opt
    test_data_list = get_test_data_list(test_opt)
    print(test_data_list)

    # # cuda config
    # # set the device for training
    # if test_opt.gpu_id == '0':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     print('USE GPU 0')

    # load model
    model = ssp()
   
    if test_opt.load is None:
        print("not found model")
    model.load_state_dict(torch.load(test_opt.load, map_location=torch.device('cpu')))
    print('load model from', test_opt.load)
    print("Start test")

    val(test_data_list, model)



