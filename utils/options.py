import argparse
import os
import torch

def str_to_list(s):
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        elements = s[1:-1].split(',')
        return [int(e.strip()) for e in elements]
    else:
        return [int(s)]

class Opt:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TrainOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # data augmentation
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0)
        parser.add_argument('--blur_sig', default=[0, 1])
        parser.add_argument('--jpg_prob', type=float, default=0)
        parser.add_argument('--jpg_method', default=['pil', 'cv2'])
        parser.add_argument('--jpg_qual', default=[90, 100])
        parser.add_argument('--CropSize', type=int,
                            default=224, help='scale images to this size')
        # train setting
        parser.add_argument('--batchsize', type=int,
                            default=64, help='input batch size')
        parser.add_argument('--choices', default=[1], type=str_to_list)
        parser.add_argument('--epoch', type=int, default=30)
        parser.add_argument('--lr', default=1e-4)
        parser.add_argument('--trainsize', type=int, default=256)
        parser.add_argument('--load', type=str,
                            default='./snapshot/sortnet/cityu.pth')
        parser.add_argument('--image_root', type=str,
                            default='/root/autodl-fs/genImage')
        parser.add_argument('--save_path', type=str,
                            default='./snapshot/sortnet/')
        parser.add_argument('--isPatch', action='store_false')
        parser.add_argument('--patch_size', default=32)
        parser.add_argument('--aug', action='store_false')
        parser.add_argument('--gpu_id', type=str, default='0')
        parser.add_argument('--log_name', default='log3.log',
                            help='rename the logfile', type=str)
        parser.add_argument('--val_interval', default=1,
                            type=int, help='val per interval')
        parser.add_argument('--val_batchsize', default=64, type=int)
        parser.add_argument('--set_dir', type=str)
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = True   # train or test
        opt.isVal = False
        # opt.classes = opt.classes.split(',')

        # # result dir, save results and opt
        # opt.results_dir = f"./results/{opt.detect_method}"
        # util.mkdir(opt.results_dir)

        if print_options:
            self.print_options(opt)

        # additional

        # opt.rz_interp = opt.rz_interp.split(',')
        # opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        # opt.jpg_method = opt.jpg_method.split(',')
        # opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        # if len(opt.jpg_qual) == 2:
        #     opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        # elif len(opt.jpg_qual) > 2:
        #     raise ValueError(
        #         "Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt

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
        'isPatch': True,
        'patch_size': 32,
        'aug': True,
        'gpu_id': '0',
        'log_name': 'log3.log',
        'val_interval': 1,
        'val_batchsize': 64,
        'test_set_dir': None,
        'print_gap': 2,
        'epochs': 30,
        #数据集文件夹名称列表
        'dataset_names': ['imagenet_ai_1203_cityu'],
        'folder_dict':{'nature':'0_real', 'ai':'1_fake'}
    }
    opt = Opt(**options)
    return opt