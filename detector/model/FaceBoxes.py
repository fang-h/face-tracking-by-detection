import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary


class CRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], dim=1)
        x = self.relu(x)
        return x


class RdclLayers(nn.Module):

    def __init__(self):
        super(RdclLayers, self).__init__()
        self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(128, 32, 1)
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(128, 32, kernel_size=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(128, 24, kernel_size=1),
            BasicConv2d(24, 32, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(128, 24, kernel_size=1),
            BasicConv2d(24, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class MsclLayers(nn.Module):

    def __init__(self):
        super(MsclLayers, self).__init__()
        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()

        self.conv3_1 = BasicConv2d(128, 128, kernel_size=1)
        self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = BasicConv2d(256, 128, kernel_size=1)
        self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        inception3_map = self.inception3(x)

        x = self.conv3_1(inception3_map)
        conv3_2_map = self.conv3_2(x)

        x = self.conv4_1(conv3_2_map)
        conv4_2_map = self.conv4_2(x)

        return inception3_map, conv3_2_map, conv4_2_map


class FaceBoxes(nn.Module):

    def __init__(self, phase, num_classes):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.rdcl_layer = RdclLayers()
        self.mscl_layer = MsclLayers()
        self.loc, self.conf = self.get_multi_boxes(self.num_classes)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        # # 初始化
        # if self.phase == 'train':
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             if m.bias is not None:
        #                 nn.init.xavier_normal_(m.weight.data)
        #                 m.bias.data.fill_(0.02)
        #             else:
        #                 m.weight.data.normal_(0, 0.01)
        #         elif isinstance(m, nn.BatchNorm2d):
        #             m.weight.data.fill_(1)
        #             m.bias.data.zero_()

    def get_multi_boxes(self, num_classes):
        loc_layers = []
        conf_layers = []

        # inception3 output
        loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=1)]
        conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=1)]

        # conv3_2 output
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        # conv4_2 output
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]

        return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)
        # return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def forward(self, x):
        loc = list()
        conf = list()

        x = self.rdcl_layer(x)
        inception3_map, conv3_2_map, conv4_2_map = self.mscl_layer(x)
        for (x, l, c) in zip([inception3_map, conv3_2_map, conv4_2_map], self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size()[0], -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size()[0], -1) for o in conf], dim=1)

        if self.phase == 'test':
            output = (loc.view(loc.size()[0], -1, 4),
                      self.softmax(conf.view(conf.size()[0], -1, self.num_classes)))
        if self.phase == 'train':
            output = (loc.view(loc.size()[0], -1, 4), conf.view(conf.size()[0], -1, self.num_classes))

        return output
