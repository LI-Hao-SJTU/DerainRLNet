###########################################################
# AUTHORS: Cheng-Hao CHEN & Hao LI
# @SJTU
# For free academic & research usage
# NOT for unauthorized commercial usage
###########################################################

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.init as init

class ResidualBlock(nn.Module):
  def __init__(self, channel_num, dilation=1, group=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=True)
    self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
    self.relu = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    xs = self.norm1(self.conv1(x))
    xs = self.relu(xs+x)
    return xs

class GCANet(nn.Module):
  def __init__(self, in_c=20, out_c=3, only_residual=True):
    super(GCANet, self).__init__()
   
    self.conv1 = nn.Conv2d(40, 32, 3, 1, 1)

    self.res1 = ResidualBlock(32, dilation=1)
    self.res2 = ResidualBlock(32, dilation=1)
    self.res3 = ResidualBlock(32, dilation=1)
    self.res4 = ResidualBlock(32, dilation=1)
    self.res5 = ResidualBlock(32, dilation=1)
    self.res6 = ResidualBlock(32, dilation=1)
    self.res7 = ResidualBlock(32, dilation=1)

    self.gate = nn.Conv2d(32 * 3, 32, 3, 1, 1, bias=True)
    
    self.deconv1 = nn.Conv2d(32, 16, 3, 1, 1)
    self.deconv2 = nn.Conv2d(16, out_c, 3, 1, 1)
    self.relu = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    y = self.relu(self.conv1(x))
    
    y1 = self.res1(y)
    y = self.res2(y)
    y = self.res3(y)
    y2 = self.res4(y)
    y = self.res5(y2)
    y = self.res6(y)
    y3 = self.res7(y)

    gates = self.relu(self.gate(torch.cat((y1, y2, y3), dim=1)))

    y = self.relu(self.deconv1(gates))
    y = torch.sigmoid(self.deconv2(y))

    return y

class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        mid_planes = int(out_planes/4)
        self.bn1 = nn.GroupNorm(num_groups=out_planes, num_channels=inter_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups=mid_planes, num_channels=out_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return torch.cat([x, out], 1)

class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        mid_planes = int(out_planes/4)
        self.bn1 = nn.GroupNorm(num_groups=out_planes, num_channels=inter_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups=mid_planes, num_channels=out_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=True)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return torch.cat([x, out], 1)

class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        mid_planes = int(out_planes/4)
        self.bn1 = nn.GroupNorm(num_groups=out_planes, num_channels=inter_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups=mid_planes, num_channels=out_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=True)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.se = SEBlock(out_planes, 6)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(out)
        return out

class conf(nn.Module):
    def __init__(self):
        super(conf, self).__init__()

        self.conv1 = nn.Conv2d(3,32,3,1,1)
        self.conv2 = BottleneckBlock(32, 32)
        self.trans_block2 = TransitionBlock(64, 32)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=32)
   
        self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1=self.relu(self.norm(self.conv1(x)))
        x1=self.conv2(x1)
        x1 = self.trans_block2(x1)
    
        residual = torch.sigmoid(self.refine3(x1))

        return residual

class conf0(nn.Module):
    def __init__(self):
        super(conf0, self).__init__()

        self.conv1 = nn.Conv2d(48,32,3,1,1)
        self.dense_block1=BottleneckBlock(32,32)
        self.trans_block1=TransitionBlock(64,32)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(32,32)
        self.trans_block2=TransitionBlock(64,32)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(32,32)
        self.trans_block3=TransitionBlock(64,32)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(32,32)
        self.trans_block4=TransitionBlock(64,32)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(64,32)
        self.trans_block5=TransitionBlock(96,32)

        self.dense_block6=BottleneckBlock(64,32)
        self.trans_block6=TransitionBlock(96,32)
        self.dense_block7=BottleneckBlock(64,32)
        self.trans_block7=TransitionBlock(96,32)
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,32)
        self.dense_block9=BottleneckBlock(32,32)
        self.trans_block9=TransitionBlock(64,32)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=32)
        self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1=self.relu(self.norm(self.conv1(x)))
        x1=self.dense_block1(x1)
        x1=self.trans_block1(x1)
        x_1=func.avg_pool2d(x1, 2)
        ###  32x32
        x2=(self.dense_block2(x_1))
        x2=self.trans_block2(x2)
        x_2=func.avg_pool2d(x2, 2)
        ### 16 X 16
        x3=(self.dense_block3(x_2))
        x3=self.trans_block3(x3)
        x_3=func.avg_pool2d(x3, 2)
        ## Classifier  ##
        
        x4=(self.dense_block4(x_3))
        x4=self.trans_block4(x4)
        x_4=func.upsample_nearest(x4, scale_factor=2)
        x_4=torch.cat([x_4,x3],1)

        x5=(self.dense_block5(x_4))
        x5=self.trans_block5(x5)
        x_5=func.upsample_nearest(x5, scale_factor=2)
        x_5=torch.cat([x_5,x2],1)

        x6=(self.dense_block6(x_5))
        x6=(self.trans_block6(x6))
        x_6=func.upsample_nearest(x6, scale_factor=2)
        x_6=torch.cat([x_6,x1],1)
        x_6=(self.dense_block7(x_6))
        x_6=(self.trans_block7(x_6))
        x_6=(self.dense_block8(x_6))
        x_6=(self.trans_block8(x_6))
        x_6=(self.dense_block9(x_6))
        x_6=(self.trans_block9(x_6))
        residual = 1.15*torch.sigmoid(self.refine3(x_6))

        return residual

class conf1(nn.Module):
    def __init__(self):
        super(conf1, self).__init__()
        self.conv1 = nn.Conv2d(6,32,3,1,1)
        self.dense_block1=BottleneckBlock(32,32)
        self.trans_block1=TransitionBlock(64,32)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(32,32)
        self.trans_block2=TransitionBlock(64,32)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(32,32)
        self.trans_block3=TransitionBlock(64,32)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(32,32)
        self.trans_block4=TransitionBlock(64,32)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(64,32)
        self.trans_block5=TransitionBlock(96,32)

        self.dense_block6=BottleneckBlock(64,32)
        self.trans_block6=TransitionBlock(96,32)
        self.dense_block7=BottleneckBlock(64,32)
        self.trans_block7=TransitionBlock(96,32)
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,32)
        self.dense_block9=BottleneckBlock(32,32)
        self.trans_block9=TransitionBlock(64,32)
        self.dense_block10=BottleneckBlock(32,32)
        self.trans_block10=TransitionBlock(64,32)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=32)
        self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1=self.relu(self.norm(self.conv1(x)))
        x1=self.dense_block1(x1)
        x1=self.trans_block1(x1)
        x_1=func.avg_pool2d(x1, 2)
        ###  32x32
        x2=(self.dense_block2(x_1))
        x2=self.trans_block2(x2)
        x_2=func.avg_pool2d(x2, 2)
        ### 16 X 16
        x3=(self.dense_block3(x_2))
        x3=self.trans_block3(x3)
        x_3=func.avg_pool2d(x3, 2)
        ## Classifier  ##
        
        x4=(self.dense_block4(x_3))
        x4=self.trans_block4(x4)
        x_4=func.upsample_nearest(x4, scale_factor=2)
        x_4=torch.cat([x_4,x3],1)

        x5=(self.dense_block5(x_4))
        x5=self.trans_block5(x5)
        x_5=func.upsample_nearest(x5, scale_factor=2)
        x_5=torch.cat([x_5,x2],1)

        x6=(self.dense_block6(x_5))
        x6=(self.trans_block6(x6))
        x_6=func.upsample_nearest(x6, scale_factor=2)
        x_6=torch.cat([x_6,x1],1)
        x_6=(self.dense_block7(x_6))
        x_6=(self.trans_block7(x_6))
        x_6=(self.dense_block8(x_6))
        x_6=(self.trans_block8(x_6))
        x_6=(self.dense_block9(x_6))
        x_6=(self.trans_block9(x_6))
        x_6=(self.dense_block10(x_6))
        x_6=(self.trans_block10(x_6))
        residual = torch.sigmoid(self.refine3(x_6))

        return residual

class tower(nn.Module):
    def __init__(self):
        super(tower, self).__init__()
        self.conv1 = nn.Conv2d(123,32,3,1,1)
        self.conv2 = BottleneckBlock(32, 32)
        self.trans_block2 = TransitionBlock(64, 32)
        self.conv3 = BottleneckBlock(32, 32)
        self.trans_block3 = TransitionBlock(64, 32)
        self.conv4 = BottleneckBlock(32, 32)
        self.trans_block4 = TransitionBlock(64, 32)
        self.conv5 = BottleneckBlock(32, 32)
        self.trans_block5 = TransitionBlock(64, 32)
        self.conv6 = BottleneckBlock(32, 32)
        self.trans_block6 = TransitionBlock(64, 32)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=32)
        
        self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1=self.relu(self.norm(self.conv1(x)))
       
        x1=self.conv2(x1)
        x1 = self.trans_block2(x1)
        x1=self.conv3(x1)
        x1 = self.trans_block3(x1)
        x1=self.conv4(x1)
        x1 = self.trans_block4(x1)
        x1=self.conv5(x1)
        x1 = self.trans_block5(x1)
        x1=self.conv6(x1)
        x1 = self.trans_block6(x1)
        residual = torch.sigmoid(self.refine3(x1))

        return residual
        
class Dense_rainall(nn.Module):
    def __init__(self):
        super(Dense_rainall, self).__init__()
        self.dense0=Dense_base_down0()
        self.dense1=Dense_base_down1()
        self.dense2=Dense_base_down2()
        self.dense00=Dense_base_down00()
        self.dense11=Dense_base_down11()
        self.dense22=Dense_base_down22()
        self.gconv=GCANet()
        self.upsample = func.upsample_nearest
        self.conv1010 = nn.Conv2d(36, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(36, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(36, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(36, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.tower1 = tower()
      
        self.convq1 = nn.Conv2d(3, 16, 1, 1, 0)
        self.convx1 = nn.Conv2d(96, 18, 1, 1, 0)
      
        self.relu=nn.LeakyReLU(0.2, inplace=True)
      
        self.upsample = func.upsample_nearest
        self.res1 = conf1()
  
        self.res3 = conf()
        self.res4 = conf()
    
        self.res33 = conf0()
        self.res44 = conf0()
        self.relu1 = nn.ReLU(inplace=True)
       
    def forward(self, x,y):
        x00=self.relu(self.convq1(x))
        xc1,xc2,xc3,xc_3=self.dense2(x00)
        xb1,xb2,xb3,xb_3=self.dense1(x00)
        xa1,xa2,xa3,xa_3=self.dense0(x00)
      
        x_3 = torch.cat([xa2,xb2,xc2],1)
        x_4 = torch.cat([xa3,xb3,xc3],1)

        x_3 = self.res33(x_3)
        x_4 = self.res44(x_4)
      
        x_223 = torch.cat([xa2,x_3,x_3,x_3],1)
        x_224 = torch.cat([xa3,x_4,x_4,x_4],1)
        
        x_333 = torch.cat([xb2,x_3,x_3,x_3],1)
        x_334 = torch.cat([xb3,x_4,x_4,x_4],1)
      
        x_443 = torch.cat([xc2,x_3,x_3,x_3],1)
        x_444 = torch.cat([xc3,x_4,x_4,x_4],1)

        xc5,xc6=self.dense22(xc_3,xc1,x_443,x_444)
        xb5,xb6=self.dense11(xb_3,xb1,x_333,x_334)
        xa5,xa6=self.dense00(xa_3,xa1,x_223,x_224)
 
        x_1=torch.cat([xa5,xb5,xc5],1)
        
        h_b = self.tower1(x_1)
       
        x_down = func.avg_pool2d(x, 2)
        h_c = torch.cat([x_down,h_b],1)
        
        h_d = self.res1(h_c)
       
        print(h_d)
        h_e = h_b-(torch.div(0.05,h_d)-0.05)*(1-2*h_b)
        h_e = self.relu1(h_e)
       
        x_6 = torch.cat([xa6,xb6,xc6],1)

        y3 = func.avg_pool2d(y, 2)
        y4 = func.avg_pool2d(y, 4)
       
        k3 = 0.15*self.res3(y3)
        k4 = 0.15*self.res4(y4)
        y_3 = y3+0.15*k3*y3
        y_4 = y4+0.15*k4*y4
        
        h_f = self.upsample(h_e,scale_factor=2)
        
        x9 = self.relu(self.convx1(x_6))
        x9 = torch.cat([x9,h_f,h_f,h_f,h_f,h_f,h_f],1)

        shape_out = x9.data.size()
        shape_out = shape_out[2:4]

        x101 = func.avg_pool2d(x9, 32)
        x102 = func.avg_pool2d(x9, 16)
        x103 = func.avg_pool2d(x9, 8)
        x104 = func.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        x10 = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
      
        x10 = self.gconv(x10)
       
        return x10,x_3,y_3,x_4,y_4,h_b,h_d,k3,k4

class Dense_base_down2(nn.Module):
    def __init__(self):
        super(Dense_base_down2, self).__init__()

        self.dense_block1=BottleneckBlock2(16,16)
        self.trans_block1=TransitionBlock(32,16)
        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock2(16,16)
        self.trans_block2=TransitionBlock(32,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock2(16,16)
        self.trans_block3=TransitionBlock(32,16)

    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)
        x_1=func.avg_pool2d(x1, 2)
        ###  32x32
        x2=(self.dense_block2(x_1))
        x2=self.trans_block2(x2)
        x_2=func.avg_pool2d(x2, 2)
        ### 16 X 16
        x3=(self.dense_block3(x_2))
        x3=self.trans_block3(x3)
        x_3=func.avg_pool2d(x3, 2)
        ## Classifier  ##

        return x1,x2,x3,x_3

class Dense_base_down22(nn.Module):
    def __init__(self):
        super(Dense_base_down22, self).__init__()
        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock2(16,16)
        self.trans_block4=TransitionBlock(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock2(41,16)
        self.trans_block5=TransitionBlock(57,16)

        self.dense_block6=BottleneckBlock2(41,16)
        self.trans_block6=TransitionBlock(57,16)

    def forward(self, x,x1,x2,x3):
        ## 256x256
       
        ## Classifier  ##
        
        x4=(self.dense_block4(x))
        x4=self.trans_block4(x4)
        x_4=func.upsample_nearest(x4, scale_factor=2)
        x_4=torch.cat([x_4,x3],1)

        x5=(self.dense_block5(x_4))
        x5=self.trans_block5(x5)
        x_5=func.upsample_nearest(x5, scale_factor=2)
        x_5=torch.cat([x_5,x2],1)

        x6=(self.dense_block6(x_5))
        x6=(self.trans_block6(x6))
        x_6=func.upsample_nearest(x6, scale_factor=2)
        x_6=torch.cat([x_6,x1],1)

        return x_5,x_6

class Dense_base_down1(nn.Module):
    def __init__(self):
        super(Dense_base_down1, self).__init__()

        self.dense_block1=BottleneckBlock1(16,16)
        self.trans_block1=TransitionBlock(32,16)
        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock1(16,16)
        self.trans_block2=TransitionBlock(32,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock1(16,16)
        self.trans_block3=TransitionBlock(32,16)

    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)
        x_1=func.avg_pool2d(x1, 2)
        ###  32x32
        x2=(self.dense_block2(x_1))
        x2=self.trans_block2(x2)
        x_2=func.avg_pool2d(x2, 2)
        ### 16 X 16
        x3=(self.dense_block3(x_2))
        x3=self.trans_block3(x3)
        x_3=func.avg_pool2d(x3, 2)
        ## Classifier  ##

        return x1,x2,x3,x_3

class Dense_base_down11(nn.Module):
    def __init__(self):
        super(Dense_base_down11, self).__init__()
        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock1(16,16)
        self.trans_block4=TransitionBlock(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock1(41,16)
        self.trans_block5=TransitionBlock(57,16)

        self.dense_block6=BottleneckBlock1(41,16)
        self.trans_block6=TransitionBlock(57,16)

    def forward(self, x,x1,x2,x3):
        ## 256x256
        x4=(self.dense_block4(x))
        x4=self.trans_block4(x4)
        x_4=func.upsample_nearest(x4, scale_factor=2)
        x_4=torch.cat([x_4,x3],1)

        x5=(self.dense_block5(x_4))
        x5=self.trans_block5(x5)
        x_5=func.upsample_nearest(x5, scale_factor=2)
        x_5=torch.cat([x_5,x2],1)

        x6=(self.dense_block6(x_5))
        x6=(self.trans_block6(x6))
        x_6=func.upsample_nearest(x6, scale_factor=2)
        x_6=torch.cat([x_6,x1],1)

        return x_5,x_6

class Dense_base_down0(nn.Module):
    def __init__(self):
        super(Dense_base_down0, self).__init__()
        self.dense_block1=BottleneckBlock(16,16)
        self.trans_block1=TransitionBlock(32,16)
        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(16,16)
        self.trans_block2=TransitionBlock(32,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(16,16)
        self.trans_block3=TransitionBlock(32,16)

    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)
        x_1=func.avg_pool2d(x1, 2)
        ###  32x32
        x2=(self.dense_block2(x_1))
        x2=self.trans_block2(x2)
        x_2=func.avg_pool2d(x2, 2)
        ### 16 X 16
        x3=(self.dense_block3(x_2))
        x3=self.trans_block3(x3)
        x_3=func.avg_pool2d(x3, 2)
        ## Classifier  ##

        return x1,x2,x3,x_3

class Dense_base_down00(nn.Module):
    def __init__(self):
        super(Dense_base_down00, self).__init__()
        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(16,16)
        self.trans_block4=TransitionBlock(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(41,16)
        self.trans_block5=TransitionBlock(57,16)

        self.dense_block6=BottleneckBlock(41,16)
        self.trans_block6=TransitionBlock(57,16)

    def forward(self, x,x1,x2,x3):
        ## 256x256
     
        ## Classifier  ##
        
        x4=(self.dense_block4(x))
        x4=self.trans_block4(x4)
        x_4=func.upsample_nearest(x4, scale_factor=2)
        x_4=torch.cat([x_4,x3],1)

        x5=(self.dense_block5(x_4))
        x5=self.trans_block5(x5)
        x_5=func.upsample_nearest(x5, scale_factor=2)
        x_5=torch.cat([x_5,x2],1)

        x6=(self.dense_block6(x_5))
        x6=(self.trans_block6(x6))
        x_6=func.upsample_nearest(x6, scale_factor=2)
        x_6=torch.cat([x_6,x1],1)

        return x_5,x_6

