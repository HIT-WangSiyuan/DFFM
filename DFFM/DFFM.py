import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

import math

import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import pdb
import numpy as np
import timm
import timm.models.vision_transformer

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        ctx.save_for_backward(p)
        return torch.bernoulli(p)

    @staticmethod
    def backward(ctx, grad_output):
        p, = ctx.saved_tensors
        # return grad_output * p * (1 - p)
        return grad_output


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, dist_token=None, *args, **kwargs):
        super(VisionTransformer, self).__init__()
        self.dist_token = dist_token

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)

        if self.dist_token is None:
            return x
        else:
            return x[:, 2:]

    def forward(self, x):
        x = self.forward_features(x)
        return x


timm.models.vision_transformer.VisionTransformer = VisionTransformer
class Discriminator(nn.Module):
    def __init__(self, max_iter=4000):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 3)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc2
        )
        #self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        #adversarial_out = self.ad_net(self.grl_layer(feature))
        adversarial_out = self.ad_net(feature)
        return adversarial_out

class ProbabilisticChannelSelector(nn.Module):
    def __init__(self, num_channels):
        super(ProbabilisticChannelSelector, self).__init__()
        self.logits = nn.Parameter(torch.randn(num_channels))

    def forward(self, x):
        probs = torch.sigmoid(self.logits)

        if self.training:
            ste = StraightThroughEstimator.apply
            selection = torch.bernoulli(probs).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            selection = selection.expand_as(x)
        else:
            selection = probs.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            selection = selection.expand_as(x)

        return x * selection
    

class Dynamic_Feature_Fusion_Model(nn.Module):
    def __init__(self, pretrained=True):
        super(Dynamic_Feature_Fusion_Model, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224',pretrained=True, pretrained_cfg_overlay=dict(file="pretrained/vit_base_patch16_224.bin"))
        #  binary CE
        self.fc = nn.Linear(768, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

        self.drop = nn.Dropout(0.3)
        self.drop2d = nn.Dropout2d(0.3)
        self.DFE_R=ProbabilisticChannelSelector(768)
        self.DFE_I=ProbabilisticChannelSelector(768)
        self.DFE_D=ProbabilisticChannelSelector(768)
        # fusion head
        self.ConvFuse = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )
        self.dis=Discriminator()

    def forward(self, x1, x2, x3):
        classtoken1 = self.vit.forward_features(x1)
        classtoken2 = self.vit.forward_features(x2)
        classtoken3 = self.vit.forward_features(x3)
        cls1=classtoken1[:,0,:]
        cls2=classtoken2[:,0,:]
        cls3=classtoken3[:,0,:]
        classtoken1 = classtoken1[:,1:,:].transpose(1, 2).view(-1, 768, 14, 14)
        classtoken2 = classtoken2[:,1:,:].transpose(1, 2).view(-1, 768, 14, 14)
        classtoken3 = classtoken3[:,1:,:].transpose(1, 2).view(-1, 768, 14, 14)
        classtoken1 = self.DFE_R(classtoken1)
        classtoken2 = self.DFE_D(classtoken2)
        classtoken3 = self.DFE_I(classtoken3)
        B, C, H, W = classtoken1.shape
        h1_temp = classtoken1.view(B, C, -1)
        h2_temp = classtoken2.view(B, C, -1)
        h3_temp = classtoken3.view(B, C, -1)

        crossh1_h2 = h2_temp @ h1_temp.transpose(-2, -1)  # [64, 768, 768]
        # pdb.set_trace()
        crossh1_h2 = F.softmax(crossh1_h2, dim=-1)
        crossedh1_h2 = (crossh1_h2 @ h1_temp).contiguous()  # [64, 768, 196]
        crossedh1_h2 = crossedh1_h2.view(B, C, H, W)

        crossh1_h3 = h3_temp @ h1_temp.transpose(-2, -1)
        crossh1_h3 = F.softmax(crossh1_h3, dim=-1)
        crossedh1_h3 = (crossh1_h3 @ h1_temp).contiguous()
        crossedh1_h3 = crossedh1_h3.view(B, C, H, W)

        # h_concat = torch.cat((classtoken1, crossedh1_h2, crossedh1_h3), dim=1)
        h_concat = classtoken1 + crossedh1_h2 + crossedh1_h3
        h_concat = self.ConvFuse(self.drop2d(h_concat))

        regmap8 = self.avgpool8(h_concat)

        logits = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))
        m1=self.dis(cls1)
        m2=self.dis(cls2)
        m3=self.dis(cls3)
        return logits,m1,m2,m3
