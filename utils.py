import numpy as np


def get_area_from_coord(x_l, x_r, y_t, y_d):
    if x_r < x_l or y_d < y_t:
        print('Wrong in Get Area')
        exit()
    return (x_r - x_l + 1) * (y_d - y_t + 1)


def get_area_from_bbox(bbox):
    """bbox:[x_l, x_r, y_t, y_d]"""
    return get_area_from_coord(bbox[0], bbox[1], bbox[2], bbox[3])


def get_wh_ratio_from_coord(x_l, x_r, y_t, y_d):
    w = x_r - x_l + 1
    h = y_d - y_t + 1
    _max = max(w, h)
    _min = min(w, h)
    return _max / float(_min)


def get_iou(bbox1, bbox2):
    """计算bbox1和bbox2的iou值,bbox：[x_left, x_right, y_top, y_down]"""
    x_min = max(bbox1[0], bbox2[0])
    x_max = min(bbox1[1], bbox2[1])
    y_min = max(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])
    inter_w = max(x_max - x_min + 1, 0)
    inter_h = max(y_max - y_min + 1, 0)
    inter_area = inter_w * inter_h
    bbox1_area = get_area_from_bbox(bbox1)
    bbox2_area = get_area_from_bbox(bbox2)
    iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
    return iou


def get_max_iou_id(id_iou):
    """id_iou:[[id0, iou0],[id1, iou1],[id2, iou2],...]"""
    iou = []
    for l1 in id_iou:
        iou.append(l1[1])
    if len(iou) > 0:
        return np.argmax(iou)
    else:
        return None


def get_ios(bbox1, bbox2):
    """计算bbox1和bbox2的inter_area和area_of_bbox1的比值
    ,bbox：[x_left, x_right, y_top, y_down]"""

    x_min = max(bbox1[0], bbox2[0])
    x_max = min(bbox1[1], bbox2[1])
    y_min = max(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])
    inter_w = max(x_max - x_min + 1, 0)
    inter_h = max(y_max - y_min + 1, 0)
    inter_area = inter_w * inter_h
    bbox1_area = get_area_from_bbox(bbox1)
    ios = inter_area / float(bbox1_area)
    return ios


def check_instance_identical_by_iou(instance1, instance2, iou_threshold):
    bbox1 = instance1.get_latest_bbox()
    bbox2 = instance2.get_latest_bbox()
    iou = get_iou(bbox1, bbox2)
    if iou > iou_threshold:
        return True
    else:
        return False


def check_bbox_identical_by_ios(bbox_ins, bbox_det, ios_threshold):
    ios = get_ios(bbox_ins, bbox_det)
    if ios > ios_threshold:
        return True
    else:
        return False


def get_sum_still(bbox1, bbox2):
    sum_still = 0
    size1 = len(bbox1)
    size2 = len(bbox2)
    if size1 != size2:
        print('Wrong size in GET_NUM_STILL')
        exit()
    for i in range(0, size1):
        sum_still += abs(bbox1[i] - bbox2[i])
    return sum_still




