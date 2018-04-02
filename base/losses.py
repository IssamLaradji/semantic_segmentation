import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np




######## CROSS ENTROPY LOSS
def crossentropy_loss(model, batch):
    model.train()
    N =  batch["images"].size(0)
    # put variables in cuda
    images = Variable(batch["images"].cuda())
    labels = Variable(batch["labels"].cuda())

    loss_func = lambda s: F.cross_entropy(s, labels, size_average=False, 
                                  ignore_index=model.ignore_index) / N
    
    if hasattr(model, "with_aux") and model.with_aux:
        scores = model.forward_aux(images, with_aux=True)
        loss = loss_func(scores["output"]) + 0.4 * loss_func(scores["aux"])
    else:
        scores = model(images)
        loss = loss_func(scores)

    
    return loss
