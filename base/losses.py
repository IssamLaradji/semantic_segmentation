import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np




######## CROSS ENTROPY LOSS
def crossentropy_loss(model, batch):
    model.train()

    # put variables in cuda
    images = Variable(batch["images"].cuda())
    labels = Variable(batch["labels"].cuda())

    scores = model(images)

    N =  batch["images"].size(0)

    loss = F.cross_entropy(scores, labels, size_average=False, 
                           ignore_index=model.ignore_index) / N

    return loss
