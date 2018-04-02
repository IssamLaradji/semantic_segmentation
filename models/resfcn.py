import torch
import torch.nn as nn
import torchvision


from . import utils_model as um
class Resnet50_8s(um.BaseModel):
    
    
    def __init__(self, train_set):

        
        super(Resnet50_8s, self).__init__(train_set)
        
        num_classes = train_set.n_classes
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()
        
        self.resnet50_32s = resnet50_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)

        # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x = self.resnet50_32s.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet50_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet50_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample_bilinear(logits_32s,
                                        size=logits_16s_spatial_dim)
        
        logits_8s += nn.functional.upsample_bilinear(logits_16s,
                                        size=logits_8s_spatial_dim)
        
        logits_upsampled = nn.functional.upsample_bilinear(logits_8s,
                                                           size=input_spatial_dim)
        
        return logits_upsampled