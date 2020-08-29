import torch.nn as nn
import torch
import torch.nn.functional as F


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
        feature_4 = self.conv1(x)
        x = self.max_pool1(feature_4)
        x = self.conv2(x)
        x = self.max_pool2(x)
        return feature_4, x


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
        self.pyramid_conf, self.pyramid_loc = self.pyramid()

    def pyramid(self):
        loc_conv = []
        conf_conv = []

        loc_conv.append(nn.Sequential(BasicConv2d(128 + 48, 64, kernel_size=1),
                                      BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 128
        loc_conv.append(nn.Sequential(BasicConv2d(128, 64, kernel_size=1),
                                      BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 64
        loc_conv.append(BasicConv2d(128, 128, kernel_size=1))  # 32
        loc_conv.append(nn.Sequential(BasicConv2d(128, 64, kernel_size=1),
                                      BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 16
        loc_conv.append(nn.Sequential(BasicConv2d(128, 64, kernel_size=1),
                                      BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 8
        loc_conv.append(nn.Sequential(BasicConv2d(128, 64, kernel_size=1),
                                      BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 4

        conf_conv.append(nn.Sequential(BasicConv2d(128 + 48, 64, kernel_size=1),
                                       BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 128
        conf_conv.append(nn.Sequential(BasicConv2d(128, 64, kernel_size=1),
                                       BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 64
        conf_conv.append(BasicConv2d(128, 128, kernel_size=1))  # 32
        conf_conv.append(nn.Sequential(BasicConv2d(128, 64, kernel_size=1),
                                       BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 16
        conf_conv.append(nn.Sequential(BasicConv2d(128, 64, kernel_size=1),
                                       BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 8
        conf_conv.append(nn.Sequential(BasicConv2d(128, 64, kernel_size=1),
                                       BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)))  # 4

        return nn.ModuleList(conf_conv), nn.ModuleList(loc_conv)

    def forward(self, x, feature_4):
        x = self.inception1(x)
        x = self.inception2(x)
        feature_32 = self.inception3(x)

        x = F.interpolate(feature_32, scale_factor=8, mode='bilinear')
        x = torch.cat([x, feature_4], dim=1)

        conf_feature = {}
        loc_feature = {}

        conf_feature['128'] = self.pyramid_conf[0](x)  # 128
        conf_feature['64'] = self.pyramid_conf[1](conf_feature['128'])  # 64
        conf_feature['32'] = self.pyramid_conf[2](feature_32)  # 32
        conf_feature['16'] = self.pyramid_conf[3](feature_32)  # 16
        conf_feature['8'] = self.pyramid_conf[4](conf_feature['16'])  # 8
        conf_feature['4'] = self.pyramid_conf[5](conf_feature['8'])  # 4

        loc_feature['128'] = self.pyramid_loc[0](x)  # 128
        loc_feature['64'] = self.pyramid_loc[1](loc_feature['128'])  # 64
        loc_feature['32'] = self.pyramid_loc[2](feature_32)  # 32
        loc_feature['16'] = self.pyramid_loc[3](feature_32)  # 16
        loc_feature['8'] = self.pyramid_loc[4](loc_feature['16'])  # 8
        loc_feature['4'] = self.pyramid_loc[5](loc_feature['8'])  # 4
        return conf_feature, loc_feature


class HRFaceBoxes(nn.Module):

    def __init__(self, phase, num_classes):
        super(HRFaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.rdcl_layer = RdclLayers()
        self.mscl_layer = MsclLayers()

        # 检测128， 64特征图上的目标
        self.conf_conv_low = nn.Conv2d(128, 3 * num_classes, kernel_size=3, padding=1, bias=False)
        self.loc_conv_low = nn.Conv2d(128, 3 * 4, kernel_size=3, padding=1, bias=False)
        # 检测32， 16， 8， 4特征图上的目标
        self.conf_conv_high = nn.Conv2d(128, 3 * num_classes, kernel_size=3, padding=1, bias=False)
        self.loc_conv_high = nn.Conv2d(128,  3 * 4, kernel_size=3, padding=1, bias=False)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        feature_4, x = self.rdcl_layer(x)
        conf_feature, loc_feature = self.mscl_layer(x, feature_4)

        loc = list()
        conf = list()
        for key in conf_feature.keys():
            if key == '128' or key == '64':
                conf.append(self.conf_conv_low(conf_feature[key]).permute(0, 2, 3, 1).contiguous())
                loc.append(self.loc_conv_low(loc_feature[key]).permute(0, 2, 3, 1).contiguous())
            else:
                conf.append(self.conf_conv_high(conf_feature[key]).permute(0, 2, 3, 1).contiguous())
                loc.append(self.loc_conv_high(loc_feature[key]).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size()[0], -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size()[0], -1) for o in conf], dim=1)

        if self.phase == 'test':
            output = (loc.view(loc.size()[0], -1, 4),
                      self.softmax(conf.view(conf.size()[0], -1, self.num_classes)))
        if self.phase == 'train':
            output = (loc.view(loc.size()[0], -1, 4), conf.view(conf.size()[0], -1, self.num_classes))

        return output




