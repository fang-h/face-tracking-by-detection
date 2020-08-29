"""get the default box in every feature maps, return types is [x, y, w, h]"""


import torch
from itertools import product as product
import math

default_anchor_sizes = [[32, 64, 128], [256], [512]]
steps = [32, 64, 128]
variance = [0.1, 0.2]
clip = False
default_anchor_sizesX = [[[19.2, 25.6, 32, 38.4, 44.8], [57.6, 76.8, 96, 115.2, 134.4]],
                         [153.6, 192, 230.4], [370.2, 384, 460.8]]


class DefaultBox(object):

    def __init__(self, img_size):
        super(DefaultBox, self).__init__()
        self.default_anchor_size = default_anchor_sizes
        self.steps = steps
        self.clip = clip
        self.img_size = img_size
        self.feature_map_size = [[math.ceil(self.img_size[0] / step), math.ceil(self.img_size[1] / step)]
                                 for step in self.steps]

    def forward(self):
        anchors = []
        for k, f_size in enumerate(self.feature_map_size):
            for i, j in product(range(f_size[0]), range(f_size[1])):
                anchor_size = self.default_anchor_size[k]
                for a_size in anchor_size:
                    w = a_size / self.img_size[1]
                    h = a_size / self.img_size[0]
                    if a_size == 32:
                        dense_cx = [x * self.steps[k] / self.img_size[1] for x
                                    in [j + 0.0, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.steps[k] / self.img_size[0] for y
                            in [i + 0.0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors.append([cx, cy, w, h])
                    elif a_size == 64:
                        dense_cx = [x * self.steps[k] / self.img_size[1] for x
                                    in [j + 0.0, j + 0.5]]
                        dense_cy = [y * self.steps[k] / self.img_size[0] for y
                                    in [i + 0.0, i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors.append([cx, cy, w, h])
                    else:
                        # in every pixel(i, j), just put one 1 anchor
                        cx = (j + 0.5) * self.steps[k] / self.img_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.img_size[0]
                        anchors.append([cx, cy, w, h])

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(min=0, max=1)
        return output


class HRDefaultBox(object):
    def __init__(self, img_size):
        super(HRDefaultBox, self).__init__()
        self.default_anchor_size = [[16, 24, 32], [40, 48, 64], [80, 96, 128], [160, 192, 256], [322, 384, 512],
                                    [640, 768, 896]]
        self.steps = [8, 16, 32, 64, 128, 256]
        self.clip = clip
        self.img_size = img_size
        self.feature_map_size = [[math.ceil(self.img_size[0] / step), math.ceil(self.img_size[1] / step)]
                                 for step in self.steps]

    def forward(self):
        anchors = []
        for k, f_size in enumerate(self.feature_map_size):
            for i, j in product(range(f_size[0]), range(f_size[1])):
                cx = (j + 0.5) * self.steps[k] / self.img_size[1]
                cy = (i + 0.5) * self.steps[k] / self.img_size[0]
                for anchor_size in self.default_anchor_size[k]:
                    w = anchor_size / self.img_size[1]
                    h = anchor_size / self.img_size[0]
                    anchors.append([cx, cy, w, h])
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(min=0, max=1)
        return output



