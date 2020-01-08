import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = SeModule(out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.nolinear1 = nn.Tanh()
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.nolinear2 = nn.Tanh()
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
            )

    def forward(self, x):
        out = self.nolinear1(self.conv1(x))
        out = self.nolinear2(self.conv2(out))
        out = self.conv3(out)
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class LinkNet(nn.Module):
    def __init__(self, in_channels=3, residual_blocks=1, init_weights=True):
        super(LinkNet, self).__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        self.block1 = nn.Sequential(
            *[Block(3, 16, 16, 16, 1) for i in range(residual_blocks)]
        )
        #  out 16
        self.conv2=Block(3, 16, 64, 24,2)
        self.block2 = nn.Sequential(
            *[Block(3, 24, 72, 24, 1) for _ in range(residual_blocks)]
        )
        #  out 24

        self.conv3=Block(5, 24, 72, 40,2)
        self.block3 = nn.Sequential(
            *[Block(5, 40, 120, 40, 1) for _ in range(residual_blocks*2)]
        )
        #  out 40

        self.conv4=Block(3, 40, 240, 80,2)
        self.block4 = nn.Sequential(
            *[Block(3, 80, 200, 80, 1) for _ in range(residual_blocks*3)]
        )
        #  out 80

        self.conv5=Block(5, 80, 672, 160,2)
        self.block5 = nn.Sequential(
            *[Block(5, 160, 960, 160, 1) for _ in range(residual_blocks*4)]
        )
        #  out 160

        # self.up1 = nn.Sequential(
        #     nn.Conv2d(16, 16, 1, 1, 0, bias=True),
        #     nn.Tanh(),
        #     # nn.Upsample(scale_factor=2 << 0, mode='bilinear')
        # )


        self.up2 = nn.Sequential(
            nn.Conv2d(24, 16, 1, 1, 0, bias=True),
            nn.Tanh(),
            nn.Upsample(scale_factor=2 << 0, mode='bilinear')
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(40, 16, 1, 1, 0, bias=True),
            nn.Tanh(),
            nn.Upsample(scale_factor=2 << 1, mode='bilinear')
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(80, 16, 1, 1, 0, bias=True),
            nn.Tanh(),
            nn.Upsample(scale_factor=2 << 2, mode='bilinear')
        )

        self.up5 = nn.Sequential(
            nn.Conv2d(160, 16, 1, 1, 0, bias=True),
            nn.Tanh(),
            nn.Upsample(scale_factor=2 << 3, mode='bilinear')
        )

        self.fusion=Block(5, 16, 48, 32,1)
        self.block_fusion = nn.Sequential(
            *[Block(5, 32, 48, 32, 1) for _ in range(residual_blocks*2)]
            # nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.final=nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)

        #  out 160

    #     self.init_params()
    #
    # def init_params(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

    def forward(self, x):
        x1=self.block1(self.conv1(x))
        x2= self.block2(self.conv2(x1))
        x3 = self.block3(self.conv3(x2))
        x4 = self.block4(self.conv4(x3))
        x5 = self.block5(self.conv5(x4))
        x=x1+self.up2(x2)+self.up3(x3)+self.up4(x4)+self.up5(x5)
        # x = torch.cat([x1, self.up2(x2), self.up3(x3), self.up4(x4), self.up5(x5)], 1)
        x = self.block_fusion(self.fusion(x))
        x=self.final(x)
        out = (torch.tanh(x) + 1) / 2
        return out
