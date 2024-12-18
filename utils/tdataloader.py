from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
from utils.patch import patch_img
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import cv2
import random as rd
from random import random, choice
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO

register_heif_opener()

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


class genImageTrainDataset(Dataset):
    def __init__(self, image_root, image_dir, opt):
        super().__init__()
        folder_map = opt.folder_dict
        nature = folder_map.get('nature', 'nature') #nature-real
        ai = folder_map.get('ai', 'ai') #ai-fake
        self.opt = opt
        self.root = os.path.join(image_root, image_dir, "train")
        self.nature_path = os.path.join(self.root, nature)
        self.nature_list = [os.path.join(self.nature_path, f) for f in os.listdir(self.nature_path)]
        self.nature_size = len(self.nature_list)
        ######
        self.ai_path = os.path.join(self.root, ai)
        self.ai_list = [os.path.join(self.ai_path, f)
                        for f in os.listdir(self.ai_path)]
        self.ai_size = len(self.ai_list)
        self.images = self.nature_list + self.ai_list
        self.labels = torch.cat(
            (torch.ones(self.nature_size), torch.zeros(self.ai_size)))

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        try:
            image = self.rgb_loader(self.images[index])
            label = self.labels[index]
        except:
            new_index = index - 1
            image = self.rgb_loader(
                self.images[max(0, new_index)])
            label = self.labels[max(0, new_index)]
        image = processing(image, self.opt)
        return image, label

    def __len__(self):
        return self.nature_size + self.ai_size


class genImageValDataset(Dataset):
    def __init__(self, image_root, image_dir, is_real, opt):
        super().__init__()
        folder_map = opt.folder_dict
        nature = folder_map.get('nature', 'nature')
        ai = folder_map.get('ai', 'ai')
        self.opt = opt
        self.root = os.path.join(image_root, image_dir, "val")
        if is_real:
            self.img_path = os.path.join(self.root, nature)
            self.img_list = [os.path.join(self.img_path, f)
                             for f in os.listdir(self.img_path)]
            self.img_len = len(self.img_list)
            self.labels = torch.ones(self.img_len)
        else:
            self.img_path = os.path.join(self.root, ai)
            self.img_list = [os.path.join(self.img_path, f)
                             for f in os.listdir(self.img_path)]
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


class genImageTestDataset(Dataset):
    def __init__(self, image_root, image_dir, opt):
        super().__init__()
        self.opt = opt
        self.root = os.path.join(image_root, image_dir, "val")
        self.nature_path = os.path.join(self.root, "nature")
        self.nature_list = [os.path.join(self.nature_path, f)
                            for f in os.listdir(self.nature_path)]
        self.nature_size = len(self.nature_list)
        self.ai_path = os.path.join(self.root, "ai")
        self.ai_list = [os.path.join(self.ai_path, f)
                        for f in os.listdir(self.ai_path)]
        self.ai_size = len(self.ai_list)
        self.images = self.nature_list + self.ai_list
        self.labels = torch.cat(
            (torch.ones(self.nature_size), torch.zeros(self.ai_size)))

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        try:
            image = self.rgb_loader(self.images[index])
            label = self.labels[index]
        except:
            new_index = index - 1
            image = self.rgb_loader(
                self.images[max(0, new_index)])
            label = self.labels[max(0, new_index)]
        image = processing(image, self.opt)
        return image, label, self.images[index]

    def __len__(self):
        return self.nature_size + self.ai_size


def get_single_loader(opt, image_dir, is_real):
    val_dataset = genImageValDataset(opt.image_root, image_dir=image_dir, is_real=is_real, opt=opt)
    val_loader = DataLoader(val_dataset, batch_size=opt.val_batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader, len(val_dataset)

def get_val_dict_infos(opt):
    dataset_names = opt.dataset_names
    val_loaders = []
    for name in dataset_names:
        datainfo = dict()
        print("val on:", name)
        datainfo['name'] = name
        datainfo['val_ai_loader'], datainfo['ai_size'] = get_single_loader(opt, datainfo['name'], is_real=False)
        datainfo['val_nature_loader'], datainfo['nature_size'] = get_single_loader(opt, datainfo['name'], is_real=True)
        val_loaders.append(datainfo)
    return val_loaders

#waste, see get_train_loader_by_names
def get_loader(opt):
    choices = opt.choices
    image_root = opt.image_root

    datasets = []
    if choices[0] == 1:
        cityu_dataset = genImageTrainDataset(image_root, "imagenet_ai_1203_cityu", opt=opt)
        datasets.append(cityu_dataset)
        print("train on: imagenet_ai_1203_cityu")

    train_dataset = torch.utils.data.ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize,
                              shuffle=True, num_workers=4, pin_memory=True)
    return train_loader

#gray replace get_loader
def get_train_loader_by_names(opt):
    image_root = opt.image_root
    dataset_names = opt.dataset_names
    datasets = []
    for name in dataset_names:
        per_data_set = genImageTrainDataset(image_root, name, opt=opt)
        datasets.append(per_data_set)
        print(f"train on: {name}")

    train_dataset = torch.utils.data.ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


def get_test_data_list(opt):
    choices = opt.choices
    test_dir = opt.test_set_dir
    list_perdict_has_loader_info = []
    loader_info_dict=dict()
    if choices[0] == 2:
        print("val on:", test_dir)
        loader_info_dict['name'] = test_dir
        loader_info_dict['val_ai_loader'], loader_info_dict['ai_size'] = get_single_loader(opt, loader_info_dict['name'], is_real=False)
        loader_info_dict['val_nature_loader'], loader_info_dict['nature_size'] = get_single_loader(opt, loader_info_dict['name'], is_real=True)
        list_perdict_has_loader_info.append(loader_info_dict)
    return list_perdict_has_loader_info

def get_test_loader(opt):
    choices = opt.choices
    image_root = opt.image_root
    datasets = []
    if choices[0] == 2:
        adm_dataset = genImageTestDataset(
            image_root, "imagenet_ai_0508_adm", opt=opt)
        datasets.append(adm_dataset)
        print("test on: imagenet_ai_0508_adm")
    if choices[1] == 2:
        biggan_dataset = genImageTestDataset(
            image_root, "imagenet_ai_0419_biggan", opt=opt)
        datasets.append(biggan_dataset)
        print("test on: imagenet_ai_0419_biggan")
    if choices[2] == 2:
        glide_dataset = genImageTestDataset(
            image_root, "imagenet_glide", opt=opt)
        datasets.append(glide_dataset)
        print("test on: imagenet_glide")
    if choices[3] == 2:
        midjourney_dataset = genImageTestDataset(
            image_root, "imagenet_midjourney", opt=opt)
        datasets.append(midjourney_dataset)
        print("test on: imagenet_midjourney")
    if choices[4] == 2:
        sdv14_dataset = genImageTestDataset(
            image_root, "imagenet_ai_0419_sdv4", opt=opt)
        datasets.append(sdv14_dataset)
        print("test on: imagenet_ai_0419_sdv4")
    if choices[5] == 2:
        sdv15_dataset = genImageTestDataset(
            image_root, "imagenet_ai_0424_sdv5", opt=opt)
        datasets.append(sdv15_dataset)
        print("test on: imagenet_ai_0424_sdv5")
    if choices[6] == 2:
        vqdm_dataset = genImageTestDataset(
            image_root, "imagenet_ai_0419_vqdm", opt=opt)
        datasets.append(vqdm_dataset)
        print("test on: imagenet_ai_0419_vqdm")
    if choices[7] == 2:
        wukong_dataset = genImageTestDataset(
            image_root, "imagenet_ai_0424_wukong", opt=opt)
        datasets.append(wukong_dataset)
        print("test on: imagenet_ai_0424_wukong")

    test_dataset = torch.utils.data.ConcatDataset(datasets)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchsize,
                             shuffle=True, num_workers=4, pin_memory=True)
    return test_loader
