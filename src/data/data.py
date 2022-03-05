import os
import glob
import random
import pickle
import skimage
from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data


class Data(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.colorMode = 'L' if args.n_colors == 1 else 'RGB'
        self.patch_size = args.patch_size
        self.quality = eval(args.quality) if isinstance(args.quality, str) else args.quality
        
        self._set_filesystem(args.dir_data)
        self.images_hr = self._scan()

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*' + self.ext)))
        return names_hr

    def _set_filesystem(self, dir_data): 
        # set basic information of dataset
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.ext = '.png'
        pass

    def __getitem__(self, idx):
        
        hr, filename = self._load_file(idx)
        if self.train:            
            ## =====data augmentation=====
            if (self.args.no_augment is False) & self.args.scale_aug:
                # it's time consuming to perform scaling augmentation on the original image
                # so, we will corp a medium-size patch from the original and them perform the augment
                hr = common.random_crop(hr, self.patch_size * 2 + 10)
                # import pdb; pdb.set_trace()
                hr = common.scale_augment(hr)

            ## crop to the patch size
            imgGT = common.random_crop(hr, self.patch_size)

            ## aug on the croped patch for saving time
            if self.args.no_augment is False:
                imgGT = common.augment_img(imgGT)
        else:
            imgGT = hr
            # pad the image to multiple of 8
            ori_h, ori_w = imgGT.shape[0:2]
            imgGT = common.padding8(imgGT, self.colorMode)
            
        
        if self.args.qm:
            imgIn, QMimg, _ = common.get_LQ(imgGT, self.quality, self.colorMode)
        elif self.args.qv:
            imgIn, _, quality = common.get_LQ(imgGT, self.quality, self.colorMode)
            h, w = imgGT.shape[0:2]
            noise_level = (100-quality)/100.0
            QMimg = np.uint8(np.ones((h, w)) * noise_level * 255)
            QMimg = QMimg[:, :, np.newaxis]
        else:
            imgIn, _, quality = common.get_LQ(imgGT, self.quality, self.colorMode)

        imgGT = np.array(imgGT, copy=False)
        imgIn = np.array(imgIn, copy=False)
        LQimg = imgIn.copy()

        if imgIn.ndim == 2:
            imgIn = imgIn[:, :, np.newaxis]
        if imgGT.ndim == 2:
            imgGT = imgGT[:, :, np.newaxis]

        if self.args.qm or self.args.qv:
            imgIn = np.concatenate((imgIn, QMimg), axis=-1)
        

        imgIn = common.uint2tensor(imgIn, self.args.rgb_range)
        imgGT = common.uint2tensor(imgGT, self.args.rgb_range)
        LQimg = common.uint2tensor(LQimg, self.args.rgb_range)


        if self.train:
            return imgIn, imgGT, filename
        else:
            return imgIn, imgGT, LQimg, (ori_h, ori_w), filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = common.imread_uint8(f_hr, mode=self.colorMode)
        return hr, filename



