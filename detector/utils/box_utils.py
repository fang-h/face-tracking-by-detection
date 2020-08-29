import torch
import numpy as np
import cv2


def point_form(boxes):
    """
    convert boxes from [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
    :param boxes:[:, 4]
    :return:
    """
    return torch.cat([boxes[:, 0:2] - boxes[:, 2:] / 2,
                      boxes[:, 0:2] + boxes[:, 2:] / 2], dim=1)


def center_form(boxes):
    """
    convert boxes from [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
    :param boxes: [:, 4]
    :return:
    """
    return torch.cat([(boxes[:, 0:2] + boxes[:, 2:]) / 2,
                      boxes[: 2:] - boxes[:, 0:2]], dim=1)


def intersect(boxes_1, boxes_2):
    """
    compute the intersect of box1 in boxes_1 and box2 in boxes_2
    :param box1:[num1, 4], the second dim is [xmin, ymin, xmax, ymax]
    :param box2:[num2, 4], the second dim is [xmin, ymin, xmax, ymax]
    :return:[num1, num2]
    """
    num1 = boxes_1.size()[0]
    num2 = boxes_2.size()[0]
    # torch.unsqueeze(add one dim)  And torch.expand(broadcast)
    max_xy = torch.min(boxes_1[:, 2:].unsqueeze(1).expand(num1, num2, 2),
                       boxes_2[:, 2:].unsqueeze(0).expand(num1, num2, 2))
    min_xy = torch.max(boxes_1[:, :2].unsqueeze(1).expand(num1, num2, 2),
                       boxes_2[:, :2].unsqueeze(0).expand(num1, num2, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]
    return inter_area


def compute_iou(boxes_1, boxes_2):
    """
    :param boxes_1: [num1, 4], the second dim is [xmin, ymin, xmax, ymax]
    :param boxes_2: [num1, 4], the second dim is [xmin, ymin, xmax, ymax]
    :return:
    """
    inter_area = intersect(boxes_1, boxes_2)
    area_1 = ((boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])
              ).unsqueeze(1).expand_as(inter_area)
    area_2 = ((boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])
              ).unsqueeze(0).expand_as(inter_area)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou


def encode(matched, default_boxes, variances):
    # difference between ground truth and prior boxes
    delta_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - default_boxes[:, :2]
    delta_cxcy /= (variances[0] * default_boxes[:, 2:])  # in SSD has no variance
    delta_wh = (matched[:, 2:] - matched[:, :2]) / default_boxes[:, 2:]
    delta_wh = torch.log(delta_wh) / variances[1]
    return torch.cat([delta_cxcy, delta_wh], dim=1)


# def new_encode(matched, default_boxes):
#     # difference between ground truth and prior boxes
#     delta_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - default_boxes[:, :2]
#     delta_cxcy /= default_boxes[:, 2:]
#     delta_wh = (matched[:, 2:] - matched[:, :2]) / default_boxes[:, 2:]
#     delta_wh = torch.log(delta_wh)
#     return torch.cat([delta_cxcy, delta_wh], dim=1)


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def decode(loc_p, default_boxes, variances):
    box_xy = loc_p[:, :2] * variances[0] * default_boxes[:, 2:] + default_boxes[:, :2]
    box_wh = torch.exp(loc_p[:, 2:] * variances[1]) * default_boxes[:, 2:]
    boxes = torch.cat([box_xy, box_wh], dim=1)
    # convert to point form
    boxes = point_form(boxes)
    return boxes


# def new_decode(loc_p, default_boxes):
#     box_xy = loc_p[:, :2] * default_boxes[:, 2:] + default_boxes[:, :2]
#     box_wh = torch.exp(loc_p[:, 2:]) * default_boxes[:, 2:]
#     boxes = torch.cat([box_xy, box_wh], dim=1)
#     # convert to point form
#     boxes = point_form(boxes)
#     return boxes


def matrix_iof(a, b):
    """a是未裁剪图片中的所有框，[num_object, 4]
    b是裁剪图片的范围，[[xmin,ymin,xmax,ymax]]"""
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)  # 保留裁剪后的有效框
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def _pad_to_square_and_resize(image, rgb_mean, size):
    """
    将图像pad成正方形，pad在右下方，pad的值是数据集的rgb均值
    """
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[np.random.randint(0, 5, 1)[0]]
    image_t = cv2.resize(image_t, (size, size), interpolation=interp_method)
    return image_t


def nms(boxes, scores, overlap=0.5):
    keep = []
    if len(boxes) == 0:
        return keep

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idx = scores.argsort()[::-1]
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        xx1 = np.maximum(x1[i], x1[idx[1:]])
        yy1 = np.maximum(y1[i], y1[idx[1:]])
        xx2 = np.minimum(x2[i], x2[idx[1:]])
        yy2 = np.minimum(y2[i], y2[idx[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        interface = w * h
        iou = interface / (area[i] + area[idx[1:]] - interface)
        index = np.where(iou <= overlap)[0]
        idx = idx[index + 1]
    return keep



























