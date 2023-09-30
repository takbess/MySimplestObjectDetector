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
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.conv1 = ConvLayer(3,8)
        self.conv2 = ConvLayer(8,16)
        self.l1 = nn.Linear(8*8*16,1024)
        self.l2 = nn.Linear(1024,1024)
        self.l3 = nn.Linear(1024,4)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(3,stride=2)

        # config 変えたいとき
        # from omegaconf import open_dict
        # with open_dict(self.cfg):
        #     self.cfg.aaa = 0

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cfg.model.one_dim == "view":
            x = x.view(x.size()[0],-1) # 16,8*8 transformed
        elif self.cfg.model.one_dim == "pool":
            x = self.pool(x)
            x = x.view(x.size()[0],-1) 

        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.l3(x) * 32 + 16
        return x


class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channel, in_channel,3,1,1)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.act(x)
        return x

class YOLOv3(nn.Module):
    def __init__(self,class_n = 80):
        super(YOLOv3, self).__init__()
        self.class_n = class_n
        self.first_block = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.residual_block_1 = self.MakeResidualBlock(64)
        self.conv_1 = nn.Conv2d(64,128,3,2,1)

        self.residual_block_2 = nn.Sequantial(self.MakeResidualBlock(128),
                                              self.MakeResidualBlock(128))
        self.conv_2 = nn.Conv2d(128,256,3,2,1)

        self.residual_block_3 = nn.Sequantial(self.MakeResidualBlock(256),self.MakeResidualBlock(256),
                                              self.MakeResidualBlock(256),self.MakeResidualBlock(256),
                                              self.MakeResidualBlock(256),self.MakeResidualBlock(256),
                                              self.MakeResidualBlock(256),self.MakeResidualBlock(256),
                                              )
        self.conv_2 = nn.Conv2d(256,512,3,2,1)

        self.residual_block_4 = nn.Sequantial(self.MakeResidualBlock(512),self.MakeResidualBlock(512),
                                              self.MakeResidualBlock(512),self.MakeResidualBlock(512),
                                              self.MakeResidualBlock(512),self.MakeResidualBlock(512),
                                              self.MakeResidualBlock(512),self.MakeResidualBlock(512),
                                              )
        self.conv_2 = nn.Conv2d(512,1024,3,2,1)

        self.residual_block_5 = nn.Sequantial(self.MakeResidualBlock(1024),self.MakeResidualBlock(1024),
                                              self.MakeResidualBlock(1024),self.MakeResidualBlock(1024),
                                              )
        self.conv_block = nn.Sequential(self.MakeResidualBlock(1024),self.MakeResidualBlock(1024),
                                        self.MakeResidualBlock(1024),self.MakeResidualBlock(1024),
                                        )
        
        self.slace3.output = nn.Conv2d(1024,(3*(4+1+self.class_n)),1,1)
        self.scale2_upsample = nn.Conv2d(1024,256,1,1)

        self.scale2_convblock = nn.Sequential(
            nn.Sequential(nn.Conv2d(768,256,1,1),
                          nn.BatchNorm2d(256),
                          nn.LeakyReLU(),
                          nn.Conv2d(256,512,3,1,1),
                          nn.BatchNorm2d(512),
                          nn.LeakyReLU(),
            ),
            self.MakeResidualBlock(512),
        )





    def MakeResidualBlock(self,fn):
        block = nn.Sequential(nn.Conv2d(fn, int(fn/2),1,1)
                              nn.BatchNorm2d(int(fn/2)),
                              nn.LeakyReLU(),
                              nn.Conv2d(int(fn/2),fn,3,1,1),
                              nn.BatchNorm2d(fn),
                              nn.LeakyReLU(),
                              )
        return block


# Usage
# model = SimpleObjectDetector()
# x = torch.randn(100,3,32,32) # RGB 32*32 image
# model(x).shape