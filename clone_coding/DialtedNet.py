import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##conv_layer 
def conv_relu(in_ch, out_ch, size=3, rate=1):
    conv_relu = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch,
                                       kernel_size=size,
                                       stride=1,
                                       padding=rate,
                                       dilation=rate),
                              nn.ReLU())
    return conv_relu

##feature extracter
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv_1 = nn.Sequential(conv_relu(3, 64, 3, 1),
                                   conv_relu(64, 64, 3, 1),
                                   nn.MaxPool2d(2, stride=2, padding=0)) #1/2
        
        self.conv_2 = nn.Sequential(conv_relu(64, 128, 3, 1),
                                   conv_relu(128, 128, 3, 1), 
                                   nn.MaxPool2d(2, stride=2, padding=0)) #1/4
        
        self.conv_3 = nn.Sequential(conv_relu(128, 256, 3, 1),
                                   conv_relu(256, 256, 3, 1),
                                   conv_relu(256, 256, 3, 1,),
                                   nn.MaxPool2d(2, stride=2, padding=0)) #1/8
        
        self.conv_4 = nn.Sequential(conv_relu(256, 512, 3, 1),
                                   conv_relu(512, 512, 3, 1), 
                                   conv_relu(512, 512, 3, 1)) 
        
        self.conv_5 = nn.Sequential(conv_relu(512, 512, 3, rate=2),
                                    conv_relu(512, 512, 3, rate=2),
                                    conv_relu(512, 512, 3, rate=2)) #dilation = 2
        
    def forward(self,x):
        print(x.size())
        out = self.conv_1(x)
        print(out.size())
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        return out
    
##classifier layer 
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=7, dilation=4, padding=12),
                                        nn.ReLU(),
                                        nn.Dropout2d(0.5),
                                        nn.Conv2d(4096,4096,kernel_size=1),#1*1 convolution
                                        nn.ReLU(),
                                        nn.Dropout2d(0.5),
                                        nn.Conv2d(4096, num_classes, kernel_size=1)
                                        ) 
    
    def forward(self, x):
        out = self.classifier(x)
        return out
    

##BasicContextModule 
#add diversifying dilation rate
"""if kernel_size=3 and dilation rate=padding, there is no size of feature change."""

class BasicContextModule(nn.Module):
    def __init__(self,num_classes):
        super(BasicContextModule, self).__init__()
        
        self.layer1 = nn.Sequential(conv_relu(num_classes, num_classes, 3, rate=1))
        self.layer2 = nn.Sequential(conv_relu(num_classes, num_classes, 3, rate=1))
        self.layer3 = nn.Sequential(conv_relu(num_classes, num_classes, 3, rate=2))
        self.layer4 = nn.Sequential(conv_relu(num_classes, num_classes, 3, rate=4))
        self.layer5 = nn.Sequential(conv_relu(num_classes, num_classes, 3, rate=8))
        self.layer6 = nn.Sequential(conv_relu(num_classes, num_classes, 3, rate=16))
        self.layer7 = nn.Sequential(conv_relu(num_classes, num_classes, 3, rate=1))
        
        #No Truncation
        """마지막 레이어에서는 필터의 dilation rate가 최대값에 도달하여 특징이 잘려나가지 않고, 원본 입력에 대한 전체적인 문맥정보를 보존"""
        self.layer8 = nn.Sequential(nn.Conv2d(num_classes, num_classes, 1, 1)) #1*1 convolution
        
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(x)
        out = self.layer4(x)
        out = self.layer5(x)
        out = self.layer6(x)
        out = self.layer7(x)
        out = self.layer8(x)
        
        return out

class DilatedNet(nn.Module):
    def __init__(self, backbone, classifier, context_module, num_classes):
        super(DilatedNet, self).__init__()
        
        self.backbone = backbone
        self.classifier = classifier
        self.context_module = context_module
        self.num_classes = num_classes
        
        #upsampling
        self.deconv = nn.ConvTranspose2d(in_channels = self.num_classes,
                                         out_channels = self.num_classes,
                                         kernel_size = 16,
                                         stride = 8,
                                         padding = 4)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.context_module(x)
        out = self.deconv(x)
        return out
    
backbone = VGG16()
num_classes = 11
classifier = Classifier(num_classes=num_classes)
context_module = BasicContextModule(num_classes=num_classes)
model = DilatedNet(backbone=backbone, classifier=classifier, context_module=context_module, num_classes=num_classes)

x = torch.randn([1, 3, 512, 512])
print("input shape : ", x.shape)
out = model(x).to(device)
print("output shape : ", out.size())

model = model.to(device)
summary(model, (3,512,512), batch_size=1, device='cpu')


    
        
        
        
        
        
        
               
        
        
                                   
        
        
    
