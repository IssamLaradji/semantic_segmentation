import os

from core import transforms as myTransforms
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import utils as ut
from torchvision import transforms


class Pascal2012(data.Dataset):
    def __init__(self, root, split, transform_name):
        self.path = root + "/VOCdevkit/"
        data_dict = make_dataset(root, split)

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




#------ aux

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
        path = os.path.join(root, 'VOCdevkit', 'VOC2012')
        img_path = path + '/JPEGImages'
        mask_path =  path + '/SegmentationClass'
        
        data_list = [l.strip('\n') for l in open(os.path.join(path,
            'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]

        ext = ".png"
    
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




class PascalPoints(Pascal2012):
    def __init__(self, root, split, transform_name):
        super().__init__(root, split, transform_name)

        self.pointsJSON = ut.jload(os.path.join( 
                                    '/mnt/datasets/public/issam/VOCdevkit/VOC2012',
                                    'whats_the_point/data', 
                                    "pascal2012_trainval_main.json"))

    def __getitem__(self, index):
        img_path = self.img_names[index]
        image = Image.open(img_path).convert('RGB')
        mask_path = self.labels[index]


        if self.split == 'train':
            label = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            label = Image.fromarray(label.astype(np.uint8))
        else:
            label = Image.open(mask_path)


        name = ut.extract_fname(img_path).split(".")[0]
        points, counts = ut.point2mask(self.pointsJSON[name], image, return_count=True, n_classes=self.n_classes-1)
        points = transforms.functional.to_pil_image(points)

        if self.joint_transform is not None:
            image, points, label = self.joint_transform([image, points, label])

        counts = torch.LongTensor(counts)
        
        return {"images":image, 
                "points":points, 
                "counts":counts,
                "index":index,
                "labels":label}

    def __len__(self):
        return len(self.img_names)
