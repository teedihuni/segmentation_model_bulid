import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()     
        self.relu = nn.ReLU(inplace=True)
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1,padding=1):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding),
                                nn.ReLU(inplace=True))
        
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
        
        #skip connection conv3
        self.skip_pool3 = nn.Conv2d(256,
                                     num_classes,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        
        #conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #1/16
        
        #skip connection conv4
        self.skip_pool4 = nn.Conv2d(512,
                                    num_classes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        
        #conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #1/32
        
        #fc6
        self.fc6 = CBR(512, 4096, 1, 1, 0)  #1*1 conv
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        #fc7
        self.fc7 = CBR(4096, 4096, 1, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        #score
        self.score_fr = nn.Conv2d(4096, num_classes, 1, 1, 0)
               
        #upscaling
        self.up16 = nn.ConvTranspose2d(num_classes, 
                                       num_classes, 
                                       kernel_size=4, 
                                       stride=2, 
                                       padding=1)
        
        self.up16to8 = nn.ConvTranspose2d(num_classes,
                                          num_classes,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1
                                          )
        
        self.up8 = nn.ConvTranspose2d(num_classes, 
                                      num_classes,
                                      kernel_size=16,
                                      stride=8,
                                      padding=4)
       

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
        pool3 = x = self.pool3(x)
        
        #skip from pool3
        skip_pool3 = self.skip_pool3(pool3)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        pool4 = x = self.pool4(x)
        
        #skip from pool4
        skip_pool4 = self.skip_pool4(pool4)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        
        x = self.score_fr(x)
        
        #up score 1
        output32 = self.up16(x)
        #sum 1        
        x = output32 + skip_pool4
        #up score 2
        output16 = self.up16to8(x)
        #sum 2
        x = output16 + skip_pool3
        #final up score 3
        final_output = self.up8(x)       

        return final_output
    
    
model = FCN8s(num_classes=21)

x = torch.randn([1, 3, 512, 512])
print("input shape : ", x.shape)
out = model(x).to(device)
print("output shape : ", out.size())

model = model.to(device)

summary(model,(3,512,512), batch_size=1, device='cpu')