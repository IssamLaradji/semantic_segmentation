import os

from . import utils_pascal as up
from base import transforms as myTransforms
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data



class Pascal2012(data.Dataset):
    def __init__(self, root, split, transform_name):
        self.path = root + "/VOCdevkit/"
        data_dict = up.make_dataset(root, split)

        self.img_names = data_dict["img_names"]
        self.labels = data_dict["labels"]

        if len(self.img_names) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.joint_transform = myTransforms.create_transformer(transform_name)
        self.split = split
        self.n_classes = 21
        self.ignore_index = 255

    def __getitem__(self, index):
        img_path = self.img_names[index]
        mask_path = self.labels[index]

        image = Image.open(img_path).convert('RGB')

        if self.split == 'train':
            label = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            label = Image.fromarray(label.astype(np.uint8))
        else:
            label = Image.open(mask_path)

        if self.joint_transform is not None:
            image, label = self.joint_transform([image, label])


        return {"images":torch.FloatTensor(image), 
                "labels":torch.LongTensor(label), 
                "index":index}

    def __len__(self):
        return len(self.img_names)