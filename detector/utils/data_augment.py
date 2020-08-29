import cv2
import numpy as np
import random
from utils.box_utils import matrix_iof


def _crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True
    if random.uniform(0, 1) > 0.5:
        for _ in range(250):
            if random.uniform(0, 1) <= 0.3:
                scale = 1
            else:
                scale = random.uniform(0.3, 1)
            short_side = min(height, width)
            w = int(scale * short_side)
            h = w

            if width == w:
                left = 0
            else:
                left = random.randrange(width - w)
            if height == h:
                top = 0
            else:
                top = random.randrange(height - h)

            roi = np.array((left, top, left + w, top + h))
            # 计算裁剪后留下的有效框的面积和原有框面积的比值
            value = matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            # 在裁剪后的图片中至少要有一个完整的框
            if not flag.any():
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            # 找到中心在裁剪范围的框
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask_a].copy()
            labels_t = labels[mask_a].copy()
            # 若不存在，则重新选择裁剪
            if boxes_t.shape[0] == 0:
                continue
            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
            # 把边缘超出裁剪范围的规划到裁剪范围，并减去裁剪带来的偏移量
            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            # 要求裁剪后的框在图像尺寸为img_dim时,至少有16个像素
            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
            if img_dim == 1024:
                min_wh = 16.0
            elif img_dim == 512:
                min_wh = 8.0
            elif img_dim == 896:
                min_wh = 16
            mask_b = np.minimum(b_w_t, b_h_t) >= min_wh
            boxes_t = boxes_t[mask_b]
            labels_t = labels_t[mask_b]

            if boxes_t.shape[0] == 0:
                continue
            pad_image_flag = False
            return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()
    p = random.uniform(0, 1)
    if p < 0.3:
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    elif p < 0.6:
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def flip(image, boxes):
    height, width, _ = image.shape
    if random.uniform(0, 1) > 0.5:
        if random.uniform(0, 1) > 0.8:
            image = cv2.flip(image, 1)
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        else:
            image = cv2.flip(image, 0)
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
    return image, boxes


def random_erasing(image, rgb_mean):
    image_t = image.copy()
    if np.random.uniform(0, 1) < 0.8:
        num_patch = np.random.randint(1, 20)
        h, w, _ = image_t.shape
        short_side = min(h, w)
        for i in range(num_patch):
            erasing_size = np.asarray([np.random.uniform(0.02, 0.08) * short_side,
                                       np.random.uniform(0.02, 0.08) * short_side])
            cut_size_half = erasing_size // 2
            center_x_min, center_x_max = cut_size_half[0], w - cut_size_half[0]
            center_y_min, center_y_max = cut_size_half[1], h - cut_size_half[1]
            center_x, center_y = np.random.randint(center_x_min, center_x_max), np.random.randint(center_y_min,
                                                                                                  center_y_max)
            x_min, y_min = center_x - int(cut_size_half[0]), center_y - int(cut_size_half[1])
            x_max, y_max = x_min + int(erasing_size[0]), y_min + int(erasing_size[1])
            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)
            image_t[y_min:y_max, x_min:x_max] = rgb_mean
    return image_t


def _pad_to_square(image, rgb_mean):
    """
    将图像pad成正方形，pad在右下方，pad的值是数据集的rgb均值
    """
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize(image, size):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (size, size), interpolation=interp_method)
    return image


def rotate_image(image, angle, rgb_means):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    (nW, nH) = (w, h)
    return cv2.warpAffine(image, M, (nW, nH), borderValue=rgb_means)


def rotate_wh(center_xy, wh, angle, cx, cy):
    '''angle = angle*pi/180
    print('angle,cos(angle)',angle,cos(angle))
    r_wh=np.abs(wh*cos(angle))'''
    s_xy = center_xy - wh / 2
    e_xy = center_xy + wh / 2
    r_s_x_s_y = rotate_xy(s_xy[:, 0], s_xy[:, 1], angle, cx, cy)
    r_s_x_e_y = rotate_xy(s_xy[:, 0], e_xy[:, 1], angle, cx, cy)
    r_e_x_s_y = rotate_xy(e_xy[:, 0], s_xy[:, 1], angle, cx, cy)
    r_e_x_e_y = rotate_xy(e_xy[:, 0], e_xy[:, 1], angle, cx, cy)
    r_x_y = np.array([r_s_x_s_y, r_s_x_e_y, r_e_x_s_y, r_e_x_e_y])

    min_xy = np.min(r_x_y, axis=0)
    max_xy = np.max(r_x_y, axis=0)
    return max_xy - min_xy


def rotate_xy(x, y, angle, cx, cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    """
    angle = angle * np.pi / 180
    x_new = (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle) + cx
    y_new = (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle) + cy
    return x_new, y_new


def rotate_angle(image, boxes, rgb_means):
    if random.uniform(0, 1) > 0.5:
        angle_list = [90, 270, 10, 350, 5, 355]  # , 15, 345]
        angle = angle_list[np.random.randint(0, len(angle_list))]
        center_xy = (boxes[:, 0:2] + boxes[:, 2:4]) / 2.0
        wh = boxes[:, 2:] - boxes[:, 0:2]
        (image_h, image_w) = image.shape[:2]
        cx, cy = image_w / 2, image_h / 2
        r_image = rotate_image(image, angle, rgb_means)
        x, y = center_xy[:, 0], center_xy[:, 1]
        r_center_xy = rotate_xy(x, y, angle, cx, cy)
        r_wh = rotate_wh(center_xy, wh, angle, cx, cy)
        lr = np.transpose(r_center_xy - r_wh / 2.0, (1, 0))
        rd = np.transpose(r_center_xy + r_wh / 2.0, (1, 0))
        boxes = np.hstack([lr, rd])
        return r_image, boxes
    else:
        return image, boxes


class DataAug(object):
    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, sample):
        image, targets = sample
        assert len(targets) > 0, "this image does not have face"
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        image_t = _distort(image_t)
        if pad_image_flag:
            image_t = _pad_to_square(image_t, rgb_mean=self.rgb_means)
        image_t, boxes_t = flip(image_t, boxes_t)
        image_t = random_erasing(image_t, self.rgb_means)
        height, width, _ = image_t.shape
        image_t = _resize(image_t, self.img_dim)
        boxes_t[:, 0::2] = boxes_t[:, 0::2] / width * 1024
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / height * 1024
        image_t, boxes_t = rotate_angle(image_t, boxes_t, self.rgb_means)
        image_t = image_t - self.rgb_means
        boxes_t[:, 0::2] = boxes_t[:, 0::2] / 1024
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / 1024
        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack([np.clip(boxes_t, a_max=1, a_min=0),  labels_t])  # [num_object, 5]
        return image_t, targets_t


class DataAug1(object):
    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, sample):
        image, targets = sample
        assert len(targets) > 0, "this image does not have face"
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        image_t = _distort(image_t)
        if pad_image_flag:
            image_t = _pad_to_square(image_t, rgb_mean=self.rgb_means)
        image_t, boxes_t = flip(image_t, boxes_t)
        image_t = random_erasing(image_t, self.rgb_means)
        height, width, _ = image_t.shape
        image_t = _resize(image_t, self.img_dim)
        boxes_t[:, 0::2] = boxes_t[:, 0::2] / width * self.img_dim
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / height * self.img_dim
        image_t, boxes_t = rotate_angle(image_t, boxes_t, self.rgb_means)
        image_t = image_t - self.rgb_means
        # 产生用于分割的map图
        semantic_label = np.zeros((self.img_dim, self.img_dim), dtype=np.long)
        for box in boxes_t:
            semantic_label[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
        boxes_t[:, 0::2] = boxes_t[:, 0::2] / self.img_dim
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / self.img_dim
        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack([np.clip(boxes_t, a_max=1, a_min=0),  labels_t])  # [num_object, 5]
        return image_t, targets_t, semantic_label


