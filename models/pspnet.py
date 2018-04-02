'''
model1: add description
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from torch.autograd import Variable
from . import utils_model as um

class PSPNet(um.BaseModel):
    def __init__(self, train_set, pretrained=True, with_aux=True):
        super().__init__()

        self.n_classes = n_classes = train_set.n_classes
        self.ignore_index = train_set.ignore_index

        # THE CNN PART
        resnet = models.resnet101(pretrained=True)
        

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
          if 'conv2' in n:
              m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
          elif 'downsample.0' in n:
              m.stride = (1, 1)

        for n, m in self.layer4.named_modules():
          if 'conv2' in n:
              m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
          elif 'downsample.0' in n:
              m.stride = (1, 1)

        self.features = nn.Sequential(self.layer0, self.layer1, 
                                      self.layer2, self.layer3)

        self.ppm = um._PyramidPoolingModule(2048, 512, (1, 2, 3, 6))


        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )

        if with_aux:
            self.aux_logits = nn.Conv2d(1024, n_classes, kernel_size=1)

        self.with_aux = True
        um.initialize_weights(self.ppm, self.final)

        # # FREEZE BATCH NORMS
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False


    def forward_aux(self, x, with_aux=False):
        x_size = x.size()

        x_layer3 = self.features(x)
        x = self.layer4(x_layer3)
        x = self.ppm(x)
        x = self.final(x)

        output = F.upsample(x, x_size[2:], mode='bilinear')

        if with_aux:
            return {"output":output,
                    "aux":F.upsample(self.aux_logits(x_layer3), 
                          x_size[2:], mode='bilinear')}
        else:
            return output

    def forward(self, x):
        return self.forward_aux(x, with_aux=False)

