import torch
import torch.nn as nn
class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        
        
        
        self.relu    = nn.ReLU(inplace=True)
        self.cnn= nn.Conv2d()
        
       

    def forward(self, x):
           
        output = self.relu(x)
        return output