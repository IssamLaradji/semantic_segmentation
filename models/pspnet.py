'''
model1: add description
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from torch.autograd import Variable


class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                #nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]

        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)

        return out

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class PSPNet(nn.Module):
  def __init__(self, n_classes, pretrained):
    super().__init__()

    self.n_classes = n_classes

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
                                  self.layer2, self.layer3,
                                  self.layer4)

    self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
    

    self.final = nn.Sequential(
        nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
        #nn.BatchNorm2d(512, momentum=.95),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(512, n_classes, kernel_size=1)
    )

    initialize_weights(self.ppm, self.final)

  def forward(self, x):
    x_size = x.size()

    x = self.features(x)
    x = self.ppm(x)
    x = self.final(x)


    return F.upsample(x, x_size[2:], mode='bilinear')

    def predict(self, batch, metric="probs"):
        self.eval()
       
        # SINGLE CLASS
        if metric == "labels":
            images = Variable(batch["images"].cuda())
            return self(images).data.max(1)[1]

        elif metric == "probs":
            images = Variable(batch["images"].cuda())
            return F.softmax(self(images),dim=1).data
    
        else:
            raise ValueError("%s not here..." % metric)