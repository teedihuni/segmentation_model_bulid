import os
import random
import torch
import torch.nn as nn
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

class Unet(nn.Module):
    def __init__(self, num_classes=11):
        super(Unet, self).__init__()
        def CBR2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers =[]
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                 kernel_size=kernel_size,stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]    
            layers += [nn.ReLU()]
            
            cbr = nn.Sequential(*layers)
            return cbr
        
        #Contracting Path
        self.encoder1_1 = CBR2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoder1_2 = CBR2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder2_1 = CBR2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoder2_2 = CBR2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder3_1 = CBR2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoder3_2 = CBR2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder4_1 = CBR2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoder4_2 = CBR2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder5_1 = CBR2D(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoder5_2 = CBR2D(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        
        #Expanding Path 
        ##(output of encoder)*(unpool + conv2d)
        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)        
        self.decoder4_2 = CBR2D(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.decoder4_1 = CBR2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.decoder3_2 = CBR2D(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.decoder3_1 = CBR2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.decoder2_2 = CBR2D(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.decoder2_1 = CBR2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        self.decoder1_2 = CBR2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.decoder1_1 = CBR2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        
        #segmentation map by 1*1 conv2d
        self.score = nn.Conv2d(in_channels=64, out_channels=num_classes,kernel_size=1,stride=1,padding=0,bias=True)
        
        
    def forward(self,x):
        encoder1_1 = self.encoder1_1(x)
        encoder1_2 = self.encoder1_2(encoder1_1)
        pool1 = self.pool1(encoder1_2)
            
        encoder2_1 = self.encoder2_1(pool1)
        encoder2_2 = self.encoder2_2(encoder2_1)
        pool2 = self.pool2(encoder2_2)  
        
        encoder3_1 = self.encoder3_1(pool2)
        encoder3_2 = self.encoder3_2(encoder3_1)
        pool3 = self.pool3(encoder3_2)
       
        encoder4_1 = self.encoder4_1(pool3)
        encoder4_2 = self.encoder4_2(encoder4_1)
        pool4 = self.pool4(encoder4_2)
                      
        encoder5_1 = self.encoder5_1(pool4)
        encoder5_2 = self.encoder5_2(encoder5_1)
                        
        unpool4 = self.unpool4(encoder5_2)
        print(encoder5_2.size(),unpool4.size(),encoder4_2.size())
        
        concat4 = torch.cat((unpool4,encoder4_2),dim=1)
        decoder4_2 = self.decoder4_2(concat4)
        decoder4_1 = self.decoder4_1(decoder4_2)
                
        unpool3 = self.unpool3(decoder4_1)
        concat3 = torch.cat((unpool3,encoder3_2),dim=1)
        decoder3_2 = self.decoder3_2(concat3)
        decoder3_1 = self.decoder3_1(decoder3_2)
        
        unpool2 = self.unpool2(decoder3_1)
        concat2 = torch.cat((unpool2,encoder2_2),dim=1)
        decoder2_2 = self.decoder2_2(concat2)
        decoder2_1 = self.decoder2_1(decoder2_2)
        
        unpool1 = self.unpool1(decoder2_1)
        concat1 = torch.cat((unpool1,encoder1_2),dim=1)
        decoder1_2 = self.decoder1_2(concat1)
        decoder1_1 = self.decoder1_1(decoder1_2)
        
        output = self.score(decoder1_1)
        return output
    
    
model = Unet(num_classes=11)
 
x = torch.randn([1,3,512,512])
print('input_shape:', x.shape)
out = model(x).to(device)
print("ouput_shape:", out.size())

model = model.to(device)

summary(model,(3,512,512), batch_size=1, device='cpu')
        
        