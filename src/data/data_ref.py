# reference based dataset, two input 
# fixed 20210710
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
from skimage import transform


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
        self.images_hr_input, self.images_hr_ref = self._scan()

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr_input)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr_input = sorted(glob.glob(os.path.join(self.dir_hr_input, '*' + self.ext)))
        names_hr_ref = sorted(glob.glob(os.path.join(self.dir_hr_ref, '*' + self.ext)))
        return names_hr_input, names_hr_ref

    def _set_filesystem(self, dir_data): 
        # set basic information of dataset
        # input,refimg
        # if refimg is same with input, just override the function,set "self.dir_hr_ref=self.dir_hr_input"
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr_input = os.path.join(self.apath, 'train/input')
        self.dir_hr_ref = os.path.join(self.apath, 'train/ref')
        self.ext = '.png'
        pass

    def __getitem__(self, idx):
        
        hr_input, hr_ref, filename = self._load_file(idx)
        if self.train:            
            ## =====data augmentation=====
            if (self.args.no_augment is False) & self.args.scale_aug:
                # it's time consuming to perform scaling augmentation on the original image
                # so, we will corp a medium-size patch from the original and them perform the augment
                hr_input, hr_ref = common.random_crop_ref(hr_input, hr_ref, self.patch_size * 2 + 10)
                # import pdb; pdb.set_trace()
                hr_input, hr_ref = common.scale_augment_ref(hr_input, hr_ref)

            ## crop to the patch size
            imgGT, hr_ref = common.random_crop_ref(hr_input, hr_ref, self.patch_size)

            ## aug on the croped patch for saving time
            if self.args.no_augment is False:
                imgGT, hr_ref = common.augment_img_ref(imgGT, hr_ref)
        else:
            imgGT = hr_input
            # pad the image to multiple of 8
            ori_h, ori_w = imgGT.shape[0:2]
            imgGT = common.padding8(imgGT, self.colorMode)
            hr_ref = common.padding8(hr_ref, self.colorMode)
            
        
        if self.args.qm:
            imgIn, QMimg, _ = common.get_LQ(imgGT, self.quality, self.colorMode)
        elif self.args.qv:
            imgIn, _, quality = common.get_LQ(imgGT, self.quality, self.colorMode)
            h, w = imgGT.shape[0:2]
            QMimg = np.uint8(np.ones((h, w)) * quality)
            QMimg = QMimg[:, :, np.newaxis]
        else:
            imgIn, _, _ = common.get_LQ(imgGT, self.quality, self.colorMode)
        
        imgGT = np.array(imgGT, copy=False)
        imgIn = np.array(imgIn, copy=False)
        LQimg = imgIn.copy()
        imgRef = np.array(hr_ref, copy=False)
        if self.name.find('_REF_INPUT') >= 0:
            imgRef = imgIn
            
        if imgIn.ndim == 2:
            imgIn = imgIn[:, :, np.newaxis]
        if imgGT.ndim == 2:
            imgGT = imgGT[:, :, np.newaxis]
        if imgRef.ndim == 2:
            imgRef = imgRef[:, :, np.newaxis]

        
        if self.args.qm or self.args.qv:
            imgIn = np.concatenate((imgIn, QMimg), axis=-1)
            if self.name.find('_REF_INPUT') >= 0:
                imgRef = imgIn
            elif self.args.qm_diff:
                QMimg_REF = np.uint8(np.ones(QMimg.shape))
                imgRef = np.concatenate((imgRef, QMimg_REF), axis=-1)
            else:
                imgRef = np.concatenate((imgRef, QMimg), axis=-1)
                
        imgIn = common.uint2tensor(imgIn, self.args.rgb_range)
        imgGT = common.uint2tensor(imgGT, self.args.rgb_range)
        LQimg = common.uint2tensor(LQimg, self.args.rgb_range)
        imgRef = common.uint2tensor(imgRef, self.args.rgb_range)


        if self.train:
            return [imgIn, imgRef], imgGT, filename
        else:
            return [imgIn, imgRef], imgGT, LQimg, (ori_h, ori_w), filename

    def __len__(self):
        if self.train:
            return len(self.images_hr_input) * self.repeat
        else:
            return len(self.images_hr_input)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr_input)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr_input = self.images_hr_input[idx]
        f_hr_ref = self.images_hr_ref[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr_input))
        hr_input = common.imread_uint8(f_hr_input, mode=self.colorMode)
        hr_ref = common.imread_uint8(f_hr_ref, mode=self.colorMode)
        if hr_input.shape != hr_ref.shape:
            hr_ref = transform.resize(hr_ref, (np.int(hr_input.shape[0]), np.int(hr_input.shape[1])), order=3)
            hr_ref = np.uint8(hr_ref * 255)
            
        return hr_input,hr_ref, filename



