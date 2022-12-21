import torch
import torch.nn as nn
class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()     
        self.relu = nn.ReLU(inplace=True)
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1,padding=1):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding),
                                nn.Relu(inplace=True))
        
        #conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #1/2
        
        #conv2
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #1/4
        
        #conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #1/8
        
        #conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #1/16
        
        #conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #1/32
        
        #fc6
        self.fc6 = nn.CBR(512, 4096, 1, 1, 0)  #1*1 conv
        self.drop6 = nn.Dropout2d()
        
        #fc7
        self.fc7 = nn.CBR(4096, 4096, 1, 1, 0)
        self.drop7 = nn.Dropout2d()
        
        #score
        self.score_fr = nn.Conv2d(4096, num_classes, 1, 1, 0)
        
        #upscaling
        self.up32 =nn.ConvTranspose2d(num_classes,num_classes, kernel_size=64, stride=32, padding=16)    
       

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        
        x = self.fc6(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = self.drop7(x)
        
        x = self.score_fr(x)
        output32 = self.up32(x)

        return output32