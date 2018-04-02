import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from torch.autograd import Variable
from . import utils_model as um

class GCN(um.BaseModel):
    def __init__(self, train_set, pretrained=True):
        super(GCN, self).__init__()

        self.n_classes = n_classes = train_set.n_classes
        self.ignore_index = train_set.ignore_index

        #self.input_size = input_size
        resnet = models.resnet152(pretrained=True)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = um._GlobalConvModule(2048, n_classes, (7, 7))
        self.gcm2 = um._GlobalConvModule(1024, n_classes, (7, 7))
        self.gcm3 = um._GlobalConvModule(512, n_classes, (7, 7))
        self.gcm4 = um._GlobalConvModule(256, n_classes, (7, 7))

        self.brm1 = um._BoundaryRefineModule(n_classes)
        self.brm2 = um._BoundaryRefineModule(n_classes)
        self.brm3 = um._BoundaryRefineModule(n_classes)
        self.brm4 = um._BoundaryRefineModule(n_classes)
        self.brm5 = um._BoundaryRefineModule(n_classes)
        self.brm6 = um._BoundaryRefineModule(n_classes)
        self.brm7 = um._BoundaryRefineModule(n_classes)
        self.brm8 = um._BoundaryRefineModule(n_classes)
        self.brm9 = um._BoundaryRefineModule(n_classes)

        um.initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
                           self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

        # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def forward(self, x):
        # if x: 512
        x_size = x.size()

        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        fs1 = self.brm5(F.upsample(gcfm1, fm3.size()[2:], mode="bilinear") + gcfm2)  # 32
        fs2 = self.brm6(F.upsample(fs1, fm2.size()[2:], mode="bilinear") + gcfm3)  # 64
        fs3 = self.brm7(F.upsample(fs2, fm1.size()[2:], mode="bilinear")+ gcfm4)  # 128
        fs4 = self.brm8(F.upsample(fs3, fm0.size()[2:], mode="bilinear"))  # 256
        out = self.brm9(F.upsample(fs4, x_size[2:], mode="bilinear"))  # 512

        return out
