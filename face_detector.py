import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2

import detector.config as config
from detector.model.FaceBoxes import FaceBoxes
from detector.model.HRFaceBoxes import HRFaceBoxes
from detector.utils.box_utils import decode, _pad_to_square_and_resize, nms
from detector.utils.default_boxes import DefaultBox, HRDefaultBox

parser = argparse.ArgumentParser(description='FaceDetector')
parser.add_argument('--FaceBoxes', default='weights/FaceBoxes.pth', type=str,
                    help='FaceBoxes trained model')
parser.add_argument('--HRFaceBoxes', default='weights/HRFaceBoxes.pth', type=str,
                    help='HRFaceBoxes trained model')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=int, help='nms threshold')
parser.add_argument('--vis_threshold', default=0.7, type=int, help='threshold for visualization')
parser.add_argument('--show_image', default=False, type=bool, help='show detect result or not')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load succeed'
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module'"""
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, cpu_option):
    if cpu_option:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if 'state_dict' in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict)
    return model


def face_detector_model(model_name):
    if model_name == 'FaceBoxes':
        model = FaceBoxes(phase='test', num_classes=config.num_classes)
        model = load_model(model, args.FaceBoxes, args.cpu)
        default_box = DefaultBox(img_size=(config.img_dim2, config.img_dim2)).forward()
    elif model_name == 'HRFaceBoxes':
        model = HRFaceBoxes(phase='test', num_classes=config.num_classes)
        model = load_model(model, args.HRFaceBoxes, args.cpu)
        default_box = HRDefaultBox(img_size=(config.img_dim2, config.img_dim2)).forward()
    return model, default_box


def face_detector(img_raw, cur_frame_counter, model, default_box):
    img = img_raw
    torch.set_grad_enabled(False)
    model.eval()
    device = torch.device('cpu' if args.cpu else 'cuda')
    model = model.to(device)

    height, width, _ = img.shape
    long_side = max(width, height)
    image_t = _pad_to_square_and_resize(img, config.rgb_mean, config.img_dim2)
    image_t = image_t - config.rgb_mean
    image_t = image_t.transpose(2, 0, 1)
    image_t = torch.from_numpy(image_t).unsqueeze(0).float()
    image_t = image_t.to(device)
    loc_p, conf_p = model(image_t)
    boxes = decode(loc_p.data.squeeze(0), default_box.data, config.variance)
    boxes = boxes * long_side
    boxes = boxes.cpu().numpy()
    scores = conf_p.squeeze(0).data.cpu().numpy()[:, -1]
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]
    keep = nms(boxes, scores, args.nms_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    outputs_useful = []
    for det in zip(boxes, scores):
        box = det[0:-1][0]
        score = det[-1]
        if score < args.vis_threshold:
            continue
        output_face = {}
        box = list(map(int, box))
        (x_l, x_r, y_t, y_d) = (box[0], box[2], box[1], box[3])
        label_str = 'face'
        output_face[label_str] = [x_l, x_r, y_t, y_d]
        outputs_useful.append(output_face)

    if args.show_image:
        for det in zip(boxes, scores):
            box = det[0:-1][0]
            score = det[-1]
            if score < args.vis_threshold:
                continue
            box = list(map(int, box))
            cv2.rectangle(img_raw, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            cv2.imwrite('detect_result/' + str(cur_frame_counter) + '.jpg', img_raw)
    return outputs_useful





