
# TODO: Some folder settings were changed compared to the original
# repository -- need to change the realtive paths for pascal voc here
# More specifically, the folder that is created after untarring the pascal
# is named VOCdevkit now instead of VOC2012

import skimage.io as io
import numpy as np
import os
import glob
import utils as ut
from scipy.io import loadmat
import cv2
from scipy.ndimage.filters import gaussian_filter 
from scipy.misc import imsave, imread
import tqdm

def make_dataset(root, split):
    assert split in ['train', 'val', 'test']
    data_dict = {"img_names": [], "labels": []}

    if split == 'train':
        path = os.path.join(root, 'VOCdevkit', 'benchmark_RELEASE', 'dataset')
        img_path =  path +'/img'
        mask_path = path +'/cls'
        data_list = [l.strip('\n') for l in open(path + '/train.txt').readlines()]
        ext = ".mat"

    elif split == 'val':
        path = os.path.join(root, 'VOCdevkit', 'VOC2012')
        img_path = path + '/JPEGImages'
        mask_path =  path + '/SegmentationClass'
        
        data_list = [l.strip('\n') for l in open(os.path.join(path,
            'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]

        ext = ".png"
           
    else:
        raise ValueError("Nope")
    
    for it in data_list:
        data_dict["img_names"] += [os.path.join(img_path, it + '.jpg')]
        data_dict["labels"] += [os.path.join(mask_path, it + '%s' % ext)]

    return data_dict


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


from PIL import Image

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
