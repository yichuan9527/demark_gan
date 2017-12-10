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



class resnet(nn.Module):
    def __init__(self, n_blocks=1, n_channels=32, num_class=79077):
        super(resnet, self).__init__()
        self.n_channels = n_channels
        self.num_class = num_class

        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)


        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.resBlocks1 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*2)  # 30*30*128
        self.resBlocks2 = self.make_block_layers(n_blocks, Residual, self.n_channels * 4)  # 30*30*128
        self.resBlocks3 = self.make_block_layers(n_blocks, Residual, self.n_channels * 4)  # 30*30*128

        self.resBlocks4 = self.make_block_layers(n_blocks, DownResidual, self.n_channels*4)  # 15*15*256
        self.resBlocks5 = self.make_block_layers(n_blocks, Residual, self.n_channels*8)  # 15*15*256
        self.resBlocks6 = self.make_block_layers(n_blocks, Residual, self.n_channels * 8)  # 15*15*256
        self.resBlocks6 = self.make_block_layers(n_blocks, Residual, self.n_channels * 8)  # 15*15*256

        self.resBlocks7 = self.make_block_layers(n_blocks, DownResidual, self.n_channels * 8)  # 8*8*512
        self.resBlocks8 = self.make_block_layers(n_blocks, Residual, self.n_channels * 16)  # 8*8*512
        self.resBlocks9 = self.make_block_layers(n_blocks, Residual, self.n_channels * 16)  # 8*8*512

        self.pooling = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, self.num_class, bias=False)

    def forward(self, input):
        # input is (nc) x 120 x 120
        output = self.Conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.Conv2(output)
        output = self.bn2(output)
        output_ = self.relu2(output)

        output = self.resBlocks1(output_)
        output = self.resBlocks2(output)
        output = self.resBlocks3(output)
        output = self.resBlocks4(output)
        output = self.resBlocks5(output)
        output = self.resBlocks6(output)
        output = self.resBlocks7(output)
        output = self.resBlocks8(output)
        output = self.resBlocks9(output)


        output = self.pooling(output)
        feature = output
        output = output.view(-1, 512)
        output = self.fc(output)

        #output = output.mean(0)
        return output, feature, output_

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








































