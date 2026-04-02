import math
import os
import random

import numpy as np
import torch.utils.data as data

import utils.utils_image as util
from utils import utils_blindsr as blindsr


class DatasetBlindSR(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetBlindSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = (opt['degradation_type'] if opt['degradation_type'] else 'bsrgan').lower()
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize * self.sf

        if self.degradation_type not in {'bsrgan', 'bsrgan_plus'}:
            raise ValueError(f'Unsupported degradation type: {self.degradation_type}')

        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_H, 'Error: H path is empty.'

    def _ensure_minimum_image_size(self, img):
        h, w = img.shape[:2]
        if h >= self.patch_size and w >= self.patch_size:
            return img

        repeat_h = max(1, math.ceil(self.patch_size / max(1, h)))
        repeat_w = max(1, math.ceil(self.patch_size / max(1, w)))
        img = np.tile(img, (repeat_h, repeat_w, 1))
        return img

    def _degrade(self, img_h):
        if self.degradation_type == 'bsrgan':
            return blindsr.degradation_bsrgan(
                img_h,
                self.sf,
                lq_patchsize=self.lq_patchsize,
                isp_model=None,
            )
        return blindsr.degradation_bsrgan_plus(
            img_h,
            self.sf,
            shuffle_prob=self.shuffle_prob,
            use_sharp=self.use_sharp,
            lq_patchsize=self.lq_patchsize,
        )

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_name = os.path.splitext(os.path.basename(H_path))[0]
        img_H = self._ensure_minimum_image_size(img_H)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W = img_H.shape[:2]

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            if 'face' in img_name.lower():
                mode = random.choice([0, 4])
            else:
                mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)

            img_H = util.uint2single(img_H)
        else:
            img_H = util.uint2single(img_H)
        img_L, img_H = self._degrade(img_H)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        return {'L': img_L, 'H': img_H, 'L_path': H_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
