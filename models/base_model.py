import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import numpy as np 
import utils_misc as ut
from skimage import morphology as morph

class BaseModel(nn.Module):
    def __init__(self, train_set):
        super().__init__()
        if hasattr(train_set, "n_classes"):
            self.n_classes = train_set.n_classes
        else:
            self.n_classes = train_set["n_classes"]
        
        if hasattr(train_set, "ignore_index"):
            self.ignore_index = train_set.ignore_index
        else:
            self.ignore_index = -100

        self.blob_mode = None



    def get_blobs(self, p_labels, return_counts=False):
        p_labels = ut.t2n(p_labels)
        n,h,w = p_labels.shape
        
        blobs = np.zeros((n, self.n_classes-1, h, w))
        counts = np.zeros((n, self.n_classes-1))
        
        # Binary case
        for i in range(n):
            for l in np.unique(p_labels[i]):
                if l == 0:
                    continue
                
                blobs[i,l-1] = morph.label(p_labels==l)
                counts[i, l-1] = (np.unique(blobs[i,l-1]) != 0).sum()

        blobs = blobs.astype(int)

        if return_counts:
            return blobs, counts

        return blobs

    def predict(self, batch, metric="probs", label=None):
        with torch.no_grad():
            self.eval()
            
            # SINGLE CLASS

            if metric == "labels":
                images = Variable(batch["images"].cuda())
                return self(images).data.max(1)[1]


            elif metric == "counts":
                labels = self.predict(batch, "labels")
                _, counts = self.get_blobs(labels, return_counts=True)

                return counts

            elif metric == "probs":
                images = Variable(batch["images"].cuda())
                return F.softmax(self(images),dim=1).data

                
            elif metric in ["blobs", "blobs_counts"]: 
                labels = self.predict(batch, "labels")

                if metric == "blobs":
                    blobs = self.get_blobs(labels)
                else:
                    blobs, counts = self.get_blobs(labels, return_counts=True)

                if self.blob_mode == "superpixels":
                    blobs = exps.get_superpixels(batch["images"], blobs)
                if metric == "blobs":
                    return blobs
                else:
                    return  blobs, counts


            elif metric == "blobs_counts":
                blobs, counts = self.get_blobs(self.predict(batch, "labels"), return_counts=True)

                return blobs, counts


            else:
                raise ValueError("%s not here..." % metric)

            

class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
                     padding=(padding,padding))

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0)