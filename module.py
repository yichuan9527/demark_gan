from __future__ import division
import torch
import torch.nn as nn
import math

# this is one block for a resnet
class Residual(nn.Module):
    def __init__(self, n_channels=64):
        super(Residual, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=1, stride=1, padding=0, bias=False)# the size will not change
        self.bn1 = nn.BatchNorm2d(self.n_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.n_channels)

    def forward(self, x):
        #input = x
        #print x.size()
        #print self.conv1
        output1 = self.conv1(x)
        output2 = self.bn1(output1)
        output3 = self.relu(output2)
        output4 = self.conv2(output3)
        output5 = self.bn2(output4)
        output = torch.add(output5, x)
        return output

class DownResidual(nn.Module):
    def __init__(self, n_channels=64):
        super(DownResidual, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(self.n_channels, self.n_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.n_channels*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.n_channels*2, self.n_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.n_channels*2)

        self.shortcut = nn.Conv2d(self.n_channels, self.n_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.shortbn = nn.BatchNorm2d(self.n_channels * 2)

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.bn1(output1)
        output3 = self.relu(output2)
        output4 = self.conv2(output3)
        output5 = self.bn2(output4)

        shortcut = self.shortcut(x)
        shortcut = self.shortbn(shortcut)
        output = torch.add(output5, shortcut)
        return output


class UpResidual(nn.Module):
    def __init__(self, n_channels=64):
        super(UpResidual, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.ConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.n_channels//2)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.n_channels//2, self.n_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.n_channels//2)

        self.shortcut1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.shortcut2 = nn.Conv2d(self.n_channels, self.n_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut3 = nn.BatchNorm2d(self.n_channels//2)



    def forward(self, x):
        shortcut1 = self.shortcut1(x)
        shortcut2 = self.shortcut2(shortcut1)
        shortcut3 = self.shortcut3(shortcut2)
        #print shortcut3.size()

        output1 = self.conv1(x)
        output2 = self.bn1(output1)
        output3 = self.relu(output2)
        output4 = self.conv2(output3)
        output5 = self.bn2(output4)
        #print output5.size()

        output = torch.add(output5, shortcut3)
        return output


class SRResNet(nn.Module):
    def __init__(self, n_channels=16, n_blocks=1):
        super(SRResNet, self).__init__()
        self.n_channels = n_channels

        self.inConv = nn.Conv2d(in_channels=3, out_channels=self.n_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.inRelu = nn.ReLU(inplace=True)
        # image is 120*120*16

        #encoder
        self.resBlocks1 = self.make_block_layers(n_blocks, DownResidual, self.n_channels)#60*60*32
        self.resBlocks2 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*2)#30*30*64
        self.resBlocks3 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*4)#15*15*128
        self.resBlocks4 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*8) #8*8*256
        self.resBlocks5 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*16) #4*4*512
        self.resBlocks6 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*32)#2*2*1024

        # FC
        self.fc1 = nn.Linear(2*2*1024, 1*1*1024)
        #self.fc_bn1 = nn.BatchNorm2d(1*1*1024)
        self.fc_relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(1*1*1024, 4*4*512)
        #self.fc_bn1 = nn.BatchNorm2d(4*4*512)
        self.fc_relu2 = nn.ReLU(inplace=True)
        # FC

        #decoder
        self.resBlocks7 = self.make_block_layers(n_blocks, UpResidual, self.n_channels*32)  # 8*8*256
        self.resBlocks8 = self.make_block_layers(n_blocks, UpResidual, self.n_channels*16)  # 16*16*128
        self.resBlocks9 = self.make_block_layers(n_blocks, UpResidual, self.n_channels*8)  # 32*32*64
        self.resBlocks10 = self.make_block_layers(n_blocks,UpResidual, self.n_channels*4)  # 64*64*32
        self.resBlocks11 = self.make_block_layers(n_blocks, UpResidual, self.n_channels*2)  # 128*128*16
        self.downsample = nn.UpsamplingBilinear2d(size=(120, 120))
        #120*120

        self.Conv1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu1 = nn.ReLU(inplace=True)

        self.outConv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0, bias=True)
        self.tan = nn.Tanh()

    def forward(self, x):
        out1 = self.inRelu(self.inConv(x))
        # encoder
        out2 = self.resBlocks1(out1)
        out3 = self.resBlocks2(out2)
        out4 = self.resBlocks3(out3)
        out5 = self.resBlocks4(out4)
        out6 = self.resBlocks5(out5)
        out7 = self.resBlocks6(out6)

        #FC
        out7 = out7.view(-1,2*2*1024)
        out7 = self.fc_relu1(self.fc1(out7))

        out7 = self.fc_relu2(self.fc2(out7))
        out7 = out7.view(-1,512,4,4)


        #decoder
        out8 = self.resBlocks7(out7)
        out9 = self.resBlocks8(out8)
        out10 = self.resBlocks9(out9)
        out11 = self.resBlocks10(out10)
        out12 = self.resBlocks11(out11)
        out13 = self.downsample(out12)

        #out
        out14 = self.relu1(self.bn1(self.Conv1(out13)))
        out15 = self.outConv(out14)
        out16 = self.tan(out15)
        #decoderr

        return out16

    def make_block_layers(self, n_blocks, block_fn, n_channel):
        layers = [block_fn(n_channels=n_channel) for x in range(n_blocks)]
        print nn.Sequential(*layers)
        return nn.Sequential(*layers)



#layers = [Residual() for x in range(15)]


class _netD(nn.Module):
    def __init__(self, n_blocks=1, n_channels=32):
        super(_netD, self).__init__()
        self.n_channels = n_channels

        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.Conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.resBlocks1 = self.make_block_layers(n_blocks, DownResidual, self.n_channels)  # 60*60*64
        self.resBlocks2 = self.make_block_layers(n_blocks, Residual, self.n_channels * 2)  # 60*60*64

        self.resBlocks3 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*2)  # 30*30*128
        self.resBlocks4 = self.make_block_layers(n_blocks, Residual, self.n_channels*4)  # 30*30*128

        self.resBlocks5 = self.make_block_layers(n_blocks, DownResidual, self.n_channels * 4)  # 15*15*256
        self.resBlocks6 = self.make_block_layers(n_blocks, Residual, self.n_channels * 8)  # 15*15*256


        self.resBlocks7 = self.make_block_layers(n_blocks, DownResidual, self.n_channels * 8)  # 8*8*512
        self.resBlocks8 = self.make_block_layers(n_blocks, Residual, self.n_channels * 16)  # 8*8*512

        self.resBlocks9 = self.make_block_layers(n_blocks, DownResidual, self.n_channels * 16)  # 4*4*1024
        self.resBlocks10 = self.make_block_layers(n_blocks, Residual, self.n_channels * 32)  # 4*4*1024

        self.pooling = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(1024, 1, bias=False)

    def forward(self, input):
        # input is (nc) x 120 x 120
        output = self.Conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.Conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.resBlocks1(output)
        output = self.resBlocks2(output)
        output = self.resBlocks3(output)
        output = self.resBlocks4(output)
        output = self.resBlocks5(output)
        output = self.resBlocks6(output)
        output = self.resBlocks7(output)
        output = self.resBlocks8(output)
        output = self.resBlocks9(output)
        output = self.resBlocks10(output)


        output = self.pooling(output)
        output = output.view(64,-1)
        output = self.fc(output)

        output = output.mean(0)
        return output.view(-1, 1)

    def make_block_layers(self, n_blocks, block_fn, n_channel):
        layers = [block_fn(n_channels=n_channel) for x in range(n_blocks)]
        print nn.Sequential(*layers)
        return nn.Sequential(*layers)



#layers = [Residual() for x in range(15)]


class localD(nn.Module):
    def __init__(self, n_blocks=1, n_channels=32):
        super(localD, self).__init__()
        self.n_channels = n_channels
        #input 70*70

        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)


        self.resBlocks1 = self.make_block_layers(n_blocks, DownResidual, self.n_channels)  # 35*35*64
        self.resBlocks2 = self.make_block_layers(n_blocks, Residual, self.n_channels * 2)  # 60*60*64

        self.resBlocks3 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*2)  # 16*16*128
        self.resBlocks4 = self.make_block_layers(n_blocks, Residual, self.n_channels*4)  # 16*16*128

        self.resBlocks5 = self.make_block_layers(n_blocks, DownResidual, self.n_channels * 4)  # 8*8*256
        self.resBlocks6 = self.make_block_layers(n_blocks, Residual, self.n_channels * 8)  # 8*8*256


        self.resBlocks7 = self.make_block_layers(n_blocks, DownResidual, self.n_channels * 8)  # 4*4*512
        self.resBlocks8 = self.make_block_layers(n_blocks, Residual, self.n_channels * 16)  # 4*4*512


        self.pooling = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(512, 1, bias=False)

    def forward(self, input):
        # input is (nc) x 120 x 120
        output = self.Conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.resBlocks1(output)
        output = self.resBlocks2(output)
        output = self.resBlocks3(output)
        output = self.resBlocks4(output)
        output = self.resBlocks5(output)
        output = self.resBlocks6(output)
        output = self.resBlocks7(output)
        output = self.resBlocks8(output)


        output = self.pooling(output)
        output = output.view(-1,512)
        output = self.fc(output)

        output = output.mean(0)
        return output.view(-1, 1)

    def make_block_layers(self, n_blocks, block_fn, n_channel):
        layers = [block_fn(n_channels=n_channel) for x in range(n_blocks)]
        print nn.Sequential(*layers)
        return nn.Sequential(*layers)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                n = (m.in_features + m.out_features) / 2
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()




























