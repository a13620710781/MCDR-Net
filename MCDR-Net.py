import torch.nn as nn
import torch

class C1(nn.Module):#4
    def __init__(self, in_channels, out_channels):
        super(C1, self).__init__()
        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn11 = nn.BatchNorm2d(32)
        self.relu11 = nn.ReLU(inplace=True)
        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)

        self.pool2 = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        # self.relu2 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn22 = nn.BatchNorm2d(32)
        self.relu22 = nn.ReLU(inplace=True)
        self.up2 = nn.ConvTranspose2d(32, 32, 4, stride=4)

        # self.conv2_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)

        self.pool3 = nn.MaxPool2d(8)
        self.conv3 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(32)
        # self.relu3 = nn.ReLU(inplace=True)
        self.conv33 = nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn33 = nn.BatchNorm2d(32)
        self.relu33 = nn.ReLU(inplace=True)
        self.up3 = nn.ConvTranspose2d(32, 32, 8, stride=8)

        self.conv4 = nn.Conv2d(96, out_channels,  kernel_size=1, stride=1, padding=0, bias=True)
        self.bn4 = nn.BatchNorm2d(out_channels)
        # self.relu4 = nn.ReLU(inplace=True)
        #
        # self.conv6 = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn6 = nn.BatchNorm2d(out_channels)
        # # self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.pool1(x)
        x1 = self.conv1(x0)
        x2 = self.bn1(x1)
        # x3 = self.relu1(x2)
        x111 = self.conv11(x2)
        x222 = self.bn11(x111)
        x333 = self.relu11(x222)
        x3_0 = self.up1(x333)

        x4_0 = self.pool2(x)
        x4 = self.conv2(x4_0)
        x5 = self.bn2(x4)
        # x6 = self.relu2(x5)
        x444 = self.conv22(x5)
        x555 = self.bn22(x444)
        x666 = self.relu22(x555)
        x6_0 = self.up2(x666)

        x7_0 = self.pool3(x)
        x7 = self.conv3(x7_0)
        x8 = self.bn3(x7)
        # x9 = self.relu3(x8)
        x777 = self.conv33(x8)
        x888 = self.bn33(x777)
        x999 = self.relu33(x888)
        x9_0 = self.up3(x999)

        e1 = torch.cat([x3_0, x6_0, x9_0], dim=1)

        x10 = self.conv4(e1)
        x11 = self.bn4(x10)
        # x12 = self.relu4(x11)

        return x11



class C2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C2, self).__init__()
        # self.pool = nn.MaxPool2d(2)
        #
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, 32, kernel_size=3, dilation=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels, 32, kernel_size=3, dilation=3, padding=3, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU(inplace=True)

        # self.conv6 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn6 = nn.BatchNorm2d(out_channels)
        # # self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        # x0 = self.pool(x)
        x0 = x
        x1 = self.conv1(x0)
        x1_1 = self.bn1(x1)
        x1_2 = self.relu1(x1_1)

        x2 = self.conv2(x0)
        x2_1 = self.bn2(x2)
        x2_2 = self.relu2(x2_1)

        x3 = self.conv3(x0)
        x3_1 = self.bn3(x3)
        x3_2 = self.relu3(x3_1)

        x4 = self.conv4(x0)
        x4_1 = self.bn4(x4)
        x4_2 = self.relu4(x4_1)

        e1 = torch.cat([x1_2, x3_2, x2_2], dim=1)

        x5 = self.conv5(e1)
        x5_1 = self.bn5(x5)
        x5_2 = self.relu5(x5_1)

        e2 = torch.cat([x4_2, x5_2], dim=1)  # 64+64+64

        # x6 = self.conv6(e2)
        # x6_1 = self.bn6(x6)
        # # x6_2 = self.relu6(x6_1)

        out = e2
        return out



class SA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SA, self).__init__()
        # self.pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(2)

        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(32)
        # self.relu4 = nn.ReLU(inplace=True)
        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)

        self.sigmoid1 = nn.Sigmoid()

        self.conv5 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn5 = nn.BatchNorm2d(out_channels)
        # self.relu5 = nn.ReLU(inplace=True)

        # self.sigmoid2 = nn.Sigmoid()
        #
        # self.conv4 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn4 = nn.BatchNorm2d(out_channels)
        # # self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = x
        x1 = self.conv1(x0)
        # x2 = self.bn1(x1)
        # x3 = self.relu1(x2)

        x4 = self.conv2(x1)
        x5 = self.bn2(x4)
        x6 = self.relu2(x5)
        x6_0 = self.pool0(x6)

        x7_0 = self.pool1(x6)
        x7 = self.conv3(x7_0)
        x8 = self.bn3(x7)
        # x9 = self.relu3(x8)
        x10 = self.conv4(x8)
        x11 = self.bn4(x10)
        # x12 = self.relu4(x11)


        x9_1 = self.sigmoid1(x11)

        s1 = x6_0 * x9_1

        x9_0 = self.up1(s1)

        x13 = self.conv5(x9_0)
        x14 = self.bn5(x13)

        d1 = x + x14

        return d1


class MCDR_Net(nn.Module):#38
    def __init__(self,in_channels,out_channels):
        super(MCDR_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(64)
        self.relu11 = nn.ReLU(inplace=True)
        self.SA1 = SA(64, 64)

        self.conv2 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(64)
        self.relu22 = nn.ReLU(inplace=True)
        # self.SA2 = SA(64, 64)

        # self.conv2_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)
        ##############################################################################################
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.C1 = C1(64, 64)
        # self.SA3 = SA(64, 64)

        #########################################################################################
        ################################################################################################\
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.C2 = C2(64, 64)
        self.SA4 = SA(64, 64)

        ###############################################################################

        # self.conv5 = nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2, bias=True)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.relu5 = nn.ReLU(inplace=True)
        #
        # self.SA1 = SA(64, 64)
        #
        # self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn6 = nn.BatchNorm2d(64)
        # self.relu6 = nn.ReLU(inplace=True)
        #
        # self.SA2 = SA(64, 64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.C3 = C1(64, 64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(inplace=True)
        self.C4 = C2(64, 64)

        self.conv7 = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(out_channels)
        # self.relu7 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        # x3 = self.relu1(x2)
        x111 = self.conv11(x2)
        x222 = self.bn11(x111)
        x333 = self.relu11(x222)
        a1 = self.SA1(x2)

        x4 = self.conv2(x)
        x5 = self.bn2(x4)
        # x6 = self.relu2(x5)
        x444 = self.conv22(x5)
        x555 = self.bn22(x444)
        x666 = self.relu22(x555)
        # a2 = self.SA2(x666)

        # x6_1 = self.conv2_1(x6)

        # d1 = x+x3
        # d2 = x+x6
        ####################################################
        x77 = self.conv3(x333)
        x88 = self.bn3(x77)
        x99 = self.relu3(x88)
        x9 = self.C1(x99)
        # a3 = self.SA3(x3)

        x11 = self.conv4(x666)
        x22 = self.bn4(x11)
        x33 = self.relu4(x22)
        x12 = self.C2(x33)
        # a4 = self.SA4(x3)
        ###############################################################

        d3 = x333 + x9 + x666
        # d3 = x3 + x9 + x6
        # a3 = self.SA3(d3)

        d4 = x666 + x12 + a1
        a4 = self.SA4(d4)

        # x14 = self.conv5(d3)
        # x15 = self.bn5(x14)
        # x16 = self.relu5(x15)
        #
        # s1 = self.SA1(x16)
        #
        # x17 = self.conv6(d4)
        # x18 = self.bn6(x17)
        # x19 = self.relu6(x18)
        #
        # s2 = self.SA2(x19)
        #
        # d5 = d3 + d4 + s1
        # d6 = d3 + d4 + s2
        #
        # # e2 = torch.cat([d5, d6], dim=1)

        x14 = self.conv5(d3)
        x15 = self.bn5(x14)
        x16 = self.relu5(x15)
        x20 = self.C3(x16)

        x17 = self.conv6(d4)
        x18 = self.bn6(x17)
        x19 = self.relu6(x18)
        x21 = self.C4(x19)

        d7 = d3 + a4 + x20
        d8 = d3 + d4 + x21
        # d8 = x9 + d4 + x21

        e3 = torch.cat([d7, d8], dim=1)

        x22 = self.conv7(e3)
        x23 = self.bn7(x22)

        return x23


if __name__ == '__main__':
    net = MCDR_Net(in_channels=1, out_channels=1)
    print(MCDR_Net)


