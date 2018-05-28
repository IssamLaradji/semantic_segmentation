import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils_misc as ut


def wtp_loss(model, batch):
    model.train()

    N = batch["images"].size(0)
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    n,k,h,w = S.size()

    S_log = F.log_softmax(O, 1)

    # IMAGE AND POINT LOSS
    # GET TARGET
    ones = torch.ones(counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones, counts], 1)
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h*w).max(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(Smax, Target, size_average=False)

    # POINT LOSS
    loss += F.nll_loss(S_log, points, 
                       ignore_index=0,
                       size_average=False)
    
    return loss / N
