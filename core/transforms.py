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
import utils_misc as ut
from misc import joint_transforms as jt
import random
import numbers
import collections
import numpy as np
from PIL import Image, ImageOps
import torch
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def create_transformer(name=""):
    if name == "Te_WTP":
       return jt.ComposeJoint(
                    [
                         [transforms.ToTensor(), None, None],
                         [transforms.Normalize(*mean_std), None, None],
                         [None, jt.ToLong(), jt.ToLong()]
                    ])


    elif name == "Tr_WTP":
       return jt.ComposeJoint(
                    [jt.RandomHorizontalFlipJoint(),            
                    [transforms.ToTensor(), None, None],
                    [transforms.Normalize(*mean_std), None, None],
                    [None, jt.ToLong(), jt.ToLong()]
                    ])


    else:
        raise ValueError("nope")

