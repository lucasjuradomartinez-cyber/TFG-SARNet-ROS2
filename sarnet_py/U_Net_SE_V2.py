# coding:utf-8
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torch.nn import Softmax
from thop import profile


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UNet(nn.Module):

    def __init__(self, n_class):
        super(UNet, self).__init__()

        self.num_resnet_layers = 152

        if self.num_resnet_layers == 50:
            resnet_raw_model = models.resnet50(pretrained=True)
            self.inplanes = 1024
        elif self.num_resnet_layers == 101:
            resnet_raw_model = models.resnet101(pretrained=True)
            self.inplanes = 1024
        elif self.num_resnet_layers == 152:
            resnet_raw_model = models.resnet152(pretrained=True)
            self.inplanes = 1024

  
    ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model.conv1
        self.encoder_rgb_bn1 = resnet_raw_model.bn1
        self.encoder_rgb_relu = resnet_raw_model.relu
        self.encoder_rgb_maxpool = resnet_raw_model.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model.layer1
        self.encoder_rgb_layer2 = resnet_raw_model.layer2
        self.encoder_rgb_layer3 = resnet_raw_model.layer3
        self.encoder_rgb_layer4 = resnet_raw_model.layer4
        
    ########  ATTENTION MECHANISM SQUEEZE AND EXITATION  ########
        self.atten_SEModule_0 = SELayer(64)
        self.atten_SEModule_1 = SELayer(256)
        self.atten_SEModule_2 = SELayer(512)
        self.atten_SEModule_3_1 = SELayer(1024)
        self.atten_SEModule_4_1 = SELayer(2048)
        
    ########  RGB DECODER  ########
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        ])
        
        self.doubleconv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )            
        ])
        
      
        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_class, kernel_size=1)
        )

    def forward(self, input):

        rgb = input
        verbose = False

        # encoder

        ######################################################################
        if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        ######################################################################
        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)
        ######################################################################
        encoder_features = [rgb]
        ######################################################################
        #rgb  = self.atten_SEModule_0(rgb)
        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (120, 160)
        rgb = self.encoder_rgb_layer1(rgb)
        encoder_features.append(rgb)
        rgb = self.atten_SEModule_1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size())  # (120, 160)
        rgb = self.encoder_rgb_layer2(rgb)
        encoder_features.append(rgb)
        rgb = self.atten_SEModule_2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (60, 80)
        rgb = self.encoder_rgb_layer3(rgb)
        encoder_features.append(rgb)
        rgb = self.atten_SEModule_3_1(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (30, 40)
        rgb = self.encoder_rgb_layer4(rgb)
        encoder_features.append(rgb)
        rgb = self.atten_SEModule_4_1(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (15, 20)
        ######################################################################
        # decoder
        ######################################################################
        deco = encoder_features[-1]
        for i, (module1, module2) in enumerate(zip(self.decoder, self.doubleconv)):
                deco = module1(deco)
                if verbose: print("decoder.size() after %d deconvolutional layer: %s" % (i+1,deco.size()))
                if verbose: print("rbg.size() of layer %d: %s" % (3-i, encoder_features[-(i + 2)].size()))
                deco = torch.cat([deco, encoder_features[-(i + 2)]], dim=1)
                if verbose: print("%d concatenation: %s" % (i+1, deco.size()))
                deco = module2(deco)
                if verbose: print("deconvolution of %d concatenation: %s" % (i+1, deco.size()))
        # Final 1x1 convolution
        deco = self.output_conv(deco)
        if verbose: print("Final:", deco.size())
        return deco


def unit_test():
    net = UNet(12).cuda(0)
    image = torch.randn(1, 3, 480, 640).cuda(0)
    with torch.no_grad():
        output = net.forward(image)
    flops, params = profile(net, inputs=(image, ))
    print(f"FLOPs: {flops}, Params: {params}")


unit_test()
