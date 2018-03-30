import json
import torch
import numpy as np
import json
import torch
import numpy as np

from torchvision import transforms
from torchvision.transforms import functional as ft
import skimage.transform as stf

from torch.autograd import Variable
from torchvision import transforms
from importlib import reload
from skimage.segmentation import mark_boundaries
from torch.utils import data
import utils as ut
import joint_transforms as jt
import random
import numbers
import collections
import numpy as np
from PIL import Image, ImageOps

import torch

def create_transformer(name=""):
    if name == "normalize":
       return jt.ComposeJoint(
                    [
                         [transforms.ToTensor(), None],
                         [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
                         [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()) ]
                    ])

    elif name == "hflipNormalize":
       return jt.ComposeJoint(
                    [jt.RandomHorizontalFlipJoint(),            
                    [transforms.ToTensor(), None],
                    [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
                    [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()) ]
                    ])

    else:
        raise ValueError("nope")

