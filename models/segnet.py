import torch
from torch import nn
from torchvision import models
from torch.autograd import Variable
from . import utils_model as um
import torch.nn.functional as F



class SegNet(um.BaseModel):
    def __init__(self, train_set, pretrained=True):
        super(SegNet, self).__init__()
        self.n_classes = n_classes = train_set.n_classes
        self.ignore_index = train_set.ignore_index

        vgg = models.vgg19(pretrained=True)

        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 4)
        )
        self.dec4 = um._DecoderBlock(1024, 256, 4)
        self.dec3 = um._DecoderBlock(512 + 256, 128, 4)
        self.dec2 = um._DecoderBlock(256 + 128, 64, 2)
        self.dec1 = um._DecoderBlock(128 + 64, n_classes, 2)
        um.initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

        # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def forward(self, x):
        x_size = x.size()
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, F.upsample(dec5, enc4.size()[2:],mode="bilinear")], 1))
        dec3 = self.dec3(torch.cat([enc3, F.upsample(dec4, enc3.size()[2:],mode="bilinear")], 1))
        dec2 = self.dec2(torch.cat([enc2, F.upsample(dec3, enc2.size()[2:],mode="bilinear")], 1))
        dec1 = self.dec1(torch.cat([enc1, F.upsample(dec2, enc1.size()[2:],mode="bilinear")], 1))


 # dec5 = self.dec5(enc5)
 #        dec4 = self.dec4(torch.cat([enc4, F.upsample(dec5, enc4.size()[2:],mode="bilinear")], 1))
 #        dec3 = self.dec3(torch.cat([enc3, F.upsample(dec4, enc3.size()[2:],mode="bilinear")], 1))
 #        dec2 = self.dec2(torch.cat([enc2, F.upsample(dec3, enc2.size()[2:],mode="bilinear")], 1))
 #        dec1 = self.dec1(torch.cat([enc1, F.upsample(dec2, enc1.size()[2:],mode="bilinear")], 1))

        return F.upsample(dec1, x_size[2:], mode="bilinear")


