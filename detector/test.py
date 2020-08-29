import os
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from model.FaceBoxes import FaceBoxes, OriginalFaceBoxes
from model.LocConf import ConfLocFaceBoxes, ConfLoc
from model.HRFaceBoxes import HRFaceBoxes
from model.SemanticFaceBoxes import SemanticFaceBoxes
from utils.default_boxes import DefaultBox, OriginalDefaultBox, ConfLocBox, HRBoxes, SemanticBoxes
from utils.box_utils import nms, decode
from loss_and_metrics.loss_and_metrics import NewMultiBoxesLoss, MultiBoxesLoss
import config
from utils.data_load import AnnotationTransform, ToTensor

GPU = [0]


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


def test(name, model_path, confidence_threshold, top_k, nms_threshold):
    cfg = config
    if name == 'OriginalFaceBoxes':
        model = OriginalFaceBoxes(phase='test', num_classes=cfg.num_classes)
        default_boxes = OriginalDefaultBox(img_size=(cfg.img_dim2, cfg.img_dim2)).forward().cuda(device=GPU[0])
        img_dim = cfg.img_dim2
    elif name == 'HRFaceBoxes':
        model = HRFaceBoxes(phase='test', num_classes=cfg.num_classes)
        default_boxes = HRBoxes(img_size=(cfg.img_dim2, cfg.img_dim2)).forward().cuda(device=GPU[0])
        img_dim = cfg.img_dim2

    cudnn.benchmark = True
    if torch.cuda.is_available():
        model = model.cuda(device=GPU[0])
        model = torch.nn.DataParallel(model, device_ids=GPU)
    if model_path is not None:
        model_dict = torch.load(os.path.join(cfg.save_path, model_path))
        # print(model_dict)
        model.load_state_dict(model_dict)

    fw = open(os.path.join(cfg.save_path, 'World_Largest_Selfie.txt'), 'w')
    #
    # criterion = NewMultiBoxesLoss(num_classes=config.num_classes, threshold_for_pos=cfg.threshold_for_pos - 0.1,
    #                               threshold_for_neg=cfg.threshold_for_neg,
    #                               ratio_between_neg_and_pos=cfg.ratio_between_neg_and_pos*1)
    # criterion = MultiBoxesLoss(config.num_classes, 0.5, 0.2, 3)

    val_images = np.load(os.path.join(os.getcwd(), 'utils', 'data_list', 'val_images.npy'), allow_pickle=True)
    val_label = np.load(os.path.join(os.getcwd(), 'utils', 'data_list', 'val_label.npy'), allow_pickle=True)
    afw_images = np.load(os.path.join(os.getcwd(), 'utils', 'data_list', 'afw.npy'), allow_pickle=True)
    pascal_images = np.load(os.path.join(os.getcwd(), 'utils', 'data_list', 'pascal.npy'), allow_pickle=True)
    model.eval()
    for i in range(len(val_images) - 3000):
        image = cv2.imread(val_images[i])
        image_name = val_images[i][-val_images[i][::-1].index('/'):]
        # image = cv2.imread('World Largest Selfie.jpg')
        # image_name = 'World Largest Selfie det.jpg'
        # print(image_name)
        # cv2.imwrite(os.path.join(cfg.save_image_path, str(i) + '.jpg'), image)   # val_images[i][59:]
        # label = val_label[i]
        # image, targets = AnnotationTransform()((image, label))
        # boxes_t = targets[:, :-1].copy()
        # labels_t = targets[:, -1].copy()
        # boxes_t[:, 0::2] = boxes_t[:, 0::2] / 1024
        # boxes_t[:, 1::2] = boxes_t[:, 1::2] / 1024
        # labels_t = np.expand_dims(labels_t, 1)
        # targets_t = np.hstack([np.clip(boxes_t, a_max=1, a_min=0), labels_t])  # [num_object, 5]
        height, width, _ = image.shape
        long_side = max(width, height)
        image_t = _pad_to_square_and_resize(image, cfg.rgb_mean, img_dim)
        image_t = image_t - cfg.rgb_mean
        image_t = image_t.transpose(2, 0, 1)
        image_t = torch.from_numpy(image_t).unsqueeze(0).float()
        # targets_t = torch.from_numpy(targets_t).float()
        # image_t = torch.cat([image_t, image_t], dim=0)
        # label = []
        # label.append(targets_t)
        # label.append(targets_t)
        if torch.cuda.is_available():
            # image_t, targets_t = image_t.cuda(device=GPU[0]), targets_t.unsqueeze(0).cuda(device=GPU[0])
            # image_t, targets_t = image_t.cuda(device=GPU[0]), [anno.cuda(device=GPU[0]) for anno in label]
            image_t = image_t.cuda(device=GPU[0])
        loc_p, conf_p = model(image_t)
        # loss_loc, loss_conf = criterion((loc_p, conf_p), targets_t, default_boxes)
        # print(loss_loc, loss_conf)
        boxes = decode(loc_p.data.squeeze(0), default_boxes.data, cfg.variance)
        boxes = boxes * long_side
        boxes = boxes.cpu().numpy()
        scores = conf_p.squeeze(0).data.cpu().numpy()[:, -1]
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]
        keep = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]

        for det in zip(boxes, scores):
            box = det[0:-1][0]
            score = det[-1]
            if score < 0.5:
                continue
            # 记录每一个框
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(image_name, score, xmin, ymin, xmax, ymax))
            # 可视化
            text = "{:.4f}".format(score)
            box = list(map(int, box))

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cx = box[0]
            cy = box[1] + 12
            cv2.putText(image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.imwrite(os.path.join(cfg.save_image_path, image_name), image)   # val_images[i][59:]
        # cv2.imwrite(image_name, image)
    fw.close()


if __name__ == '__main__':
    test('OriginalFaceBoxes', 'OriginalFaceBoxes_87.pth', 0.3, 200, 0.3)
    # test('HRFaceBoxes', 'HRFaceBoxes_147.pth', 0.3, 10000, 0.3)











