import torch
import torch.nn as nn 

class ConvLayer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
    def forward(self,x):
        x = self.pool(self.act(self.conv(x)))
        return x



class SimpleObjectDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3,8)
        self.conv2 = ConvLayer(8,16)
        self.l1 = nn.Linear(8*8*16,1024)
        self.l2 = nn.Linear(1024,1024)
        self.l3 = nn.Linear(1024,4)
        self.act = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0],-1) # 16,8*8 transformed
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.l3(x) * 32 + 16
        return x

# Usage
# model = SimpleObjectDetector()
# x = torch.randn(100,3,32,32) # RGB 32*32 image
# model(x).shape



