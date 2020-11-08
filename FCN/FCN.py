import torch
from torch import nn
from torchvision.models import vgg19

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # num_classes:训练数据的类别
        self.num_classes = num_classes
        model_vgg19 = vgg19(pretrained=True)
        ## 不使用vgg19网络中的后面的AdaptiveAvgPool2d和Linear层
        self.base_model = model_vgg19.features
        ## 定义几个需要的层操作，并且使用转置卷积将特征映射进行升维
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2,
                                          padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
    
        ## vgg19中MaxPool2d所在的层
        self.layers = {"4": "maxpool_1","9": "maxpool_2", 
                       "18": "maxpool_3", "27": "maxpool_4", 
                       "36": "maxpool_5"}

    def forward(self, x):
        output = {}
        nametag = 0
        for i, (name, layer) in enumerate(self.base_model._modules.items()):
            ## 从第一层开始获取图像的特征
            x = layer(x)
            ## 如果是layers参数指定的特征，那就保存到output中
            if name in self.layers:
                output[self.layers[name]] = x
        x5 = output["maxpool_5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["maxpool_4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["maxpool_3"]  # size=(N, 256, x.H/8,  x.W/8)
        ## 对特征进行相关的转置卷积操作,逐渐将图像放大到原始图像大小
        # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv1(x5))
        # 对应的元素相加, size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  
        # size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv2(score)) 
        # 对应的元素相加, size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score))) 
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        # size=(N, 32, x.H, x.W)
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    

        return score  # size=(N, n_class, x.H/1, x.W/1)