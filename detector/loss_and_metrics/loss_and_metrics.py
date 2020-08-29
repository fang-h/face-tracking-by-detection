import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import compute_iou, point_form, encode, log_sum_exp, new_encode


def match(ground_truth, class_label, default_boxes, threshold_for_match, threshold_for_hard_gt,
          variances, loc_t, conf_t, batch_idx):
    """
    :param ground_truth: [:, 4]
    :param class_label: [:]
    :param default_boxes: [:, 4]
    :param threshold_for_match:
    :param threshold_for_hard_gt:
    :param variances:
    :param loc_t: [:, :, 4]
    :param conf_t: [:, :]
    :param batch_idx:
    :return:
    """
    # 首先找到每一个ground truth对应的最高iou的default box，并去除掉一些困难样本

    # 计算每一个ground truth和每一个default box的iou
    iou = compute_iou(ground_truth, point_form(default_boxes))  # shape is [num_object, num_default_boxes]
    # print(iou.shape, torch.max(iou))
    # 找到每一个ground truth所对应的那个iou最大的default box
    gt_highest_default_overlap, gt_highest_default_idx = iou.max(dim=1, keepdim=True)  # shape:[num_object, 1]
    valid_gt_idx = gt_highest_default_overlap[:, 0] >= threshold_for_hard_gt
    # 记录每一个有效的ground truth所对应的那个default box的索引
    valid_gt_highest_default_idx = gt_highest_default_idx[valid_gt_idx, :]
    if valid_gt_highest_default_idx.shape[0] == 0:
        loc_t[batch_idx] = 0
        conf_t[batch_idx] = 0
        return

    #
    # 找到每一个default box对应的那个iou最大的ground truth
    default_highest_gt_overlap, default_highest_gt_idx = iou.max(dim=0, keepdim=True)
    default_highest_gt_idx.squeeze_(0)  # 去掉第0个维度
    default_highest_gt_overlap.squeeze_(0)  # 去掉第0个维度
    gt_highest_default_idx.squeeze_(1)  # 去掉第1个维度变
    valid_gt_highest_default_idx.squeeze_(1)  # 去掉第1个维度
    default_highest_gt_overlap.index_fill_(0, valid_gt_highest_default_idx, 1)
    # 对于第j个ground truth，其对应第gt_highest_default_idx[j]个default box
    # 即第gt_highest_default_idx[j]个default box，其对应第j个ground truth
    for j in range(gt_highest_default_idx.size()[0]):
        default_highest_gt_idx[gt_highest_default_idx[j]] = j

    matches = ground_truth[default_highest_gt_idx]
    conf = class_label[default_highest_gt_idx]
    conf[default_highest_gt_overlap < threshold_for_match] = 0

    loc = encode(matches, default_boxes, variances)
    loc_t[batch_idx] = loc
    conf_t[batch_idx] = conf


class MultiBoxesLoss(nn.Module):

    def __init__(self, num_classes, threshold_for_match, threshold_for_hard_gt, ratio_between_neg_and_pos):
        """almost all the parameter is used for matching"""
        super(MultiBoxesLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold_for_match = threshold_for_match
        self.threshold_for_hard_gt = threshold_for_hard_gt
        self.ratio_between_neg_and_pos = ratio_between_neg_and_pos
        self.variances = [0.1, 0.2]

    def forward(self, predictions, targets, default_boxes):
        """
        :param predictions: ([batch_size, :, 4], [batch_size, :, 2])
        :param targets: [batch_size, :, 5]
        :param default_boxes: [:, 5]
        :return:
        """
        loc_p, conf_p = predictions
        batch_size = loc_p.size()[0]
        num_default_boxes = default_boxes.size()[0]
        # loc_t and conf_t is the target of FaceBoxes's output
        loc_t = torch.Tensor(batch_size, num_default_boxes, 4)
        conf_t = torch.LongTensor(batch_size, num_default_boxes)
        # 依次处理batch中的每一个图片
        for batch_idx in range(batch_size):
            ground_truth = targets[batch_idx][:, :-1].data  # [:, 4], [xmin, ymin, xmax, ymax]
            class_label = targets[batch_idx][:, -1].data  # [:]
            default = default_boxes.data   # [:, 4]  [cx, cy, w, h]
            match(ground_truth, class_label, default, self.threshold_for_match, self.threshold_for_hard_gt,
                  self.variances, loc_t, conf_t, batch_idx)
        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        # loc_t = loc_t
        # conf_t = conf_t
        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_p)
        loc_p = loc_p[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # for hard negative mining
        batch_conf = conf_p.view(-1, self.num_classes)
        loss_conf = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_conf[pos.view(-1, 1)] = 0  # 去除掉正样本
        loss_conf = loss_conf.view((batch_size, -1))
        _, loss_conf_idx = loss_conf.sort(dim=1, descending=True)
        _, idx_rank = loss_conf_idx.sort(dim=1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.ratio_between_neg_and_pos * num_pos, max=pos.size(1) - 2) + 1
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(dim=2).expand_as(conf_p)
        neg_idx = neg.unsqueeze(dim=2).expand_as(conf_p)
        conf_p = conf_p[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # print(targets_weighted.shape)
        loss_conf = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        # print(loss_loc, loss_conf)
        loss_loc = loss_loc / N
        loss_conf = loss_conf / N

        return loss_loc, loss_conf


def new_match(ground_truth, class_label, default_boxes, threshold_for_pos, threshold_for_neg,
              loc_t, conf_t, batch_idx):
    """
    :param ground_truth: [:, 4]
    :param class_label: [:]
    :param default_boxes: [:, 4]
    :param threshold_for_pos:
    :param threshold_for_neg:
    :param loc_t: [:, :, 4]
    :param conf_t: [:, :]
    :param batch_idx:
    :return:
    """
    # 计算每一个ground truth和每一个default box的iou
    iou = compute_iou(ground_truth, point_form(default_boxes))  # shape is [num_object, num_default_boxes]
    # # print("iou:", torch.max(iou))
    # gt_highest_default_overlap, gt_highest_default_idx = iou.max(dim=1, keepdim=True)  # shape:[num_object, 1]
    # valid_gt_idx = gt_highest_default_overlap[:, 0] >= 0.2
    # # 记录每一个有效的ground truth所对应的那个default box的索引
    # valid_gt_highest_default_idx = gt_highest_default_idx[valid_gt_idx, :]
    # if valid_gt_highest_default_idx.shape[0] == 0:
    #     loc_t[batch_idx] = 0
    #     conf_t[batch_idx] = 0
    #     return

    # # 找到每一个ground truth所对应的那个iou最大的default box
    gt_highest_default_overlap, gt_highest_default_idx = iou.max(dim=1, keepdim=True)  # shape:[num_object, 1]
    # 找到每一个default box对应的那个iou最大的ground truth
    default_highest_gt_overlap, default_highest_gt_idx = iou.max(dim=0, keepdim=True)

    default_highest_gt_idx.squeeze_(0)  # 去掉第0个维度
    default_highest_gt_overlap.squeeze_(0)  # 去掉第0个维度
    gt_highest_default_idx.squeeze_(1)  # 去掉第1个维度变
    gt_highest_default_overlap.squeeze_(1)
    # 对于第j个ground truth，其对应第gt_highest_default_idx[j]个default box
    # 即第gt_highest_default_idx[j]个default box，其对应第j个ground truth
    for j in range(gt_highest_default_idx.size()[0]):
        # 当ground truth对应的iou最大default box的iou值大于匹配为正样本的阈值时，才双向匹配
        if gt_highest_default_overlap[j] >= threshold_for_pos:
            default_highest_gt_idx[gt_highest_default_idx[j]] = j

    # 一个ground truth可以由多个default box匹配，只要这些default box与该ground truth 的iou值大于匹配为正样本的阈值
    matches = ground_truth[default_highest_gt_idx]
    conf = class_label[default_highest_gt_idx]  #
    conf[default_highest_gt_overlap < threshold_for_pos] = 0.5  # conf值为0.5的表示忽略
    conf[default_highest_gt_overlap < threshold_for_neg] = 0  # conf值为0表示为负样本

    loc = encode(matches, default_boxes, [0.1, 0.2])
    loc_t[batch_idx] = loc
    conf_t[batch_idx] = conf


class NewMultiBoxesLoss(nn.Module):

    def __init__(self, num_classes, threshold_for_pos, threshold_for_neg, ratio_between_neg_and_pos):
        """almost all the parameter is used for matching"""
        super(NewMultiBoxesLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold_for_pos = threshold_for_pos
        self.threshold_for_neg = threshold_for_neg
        self.ratio_between_neg_and_pos = ratio_between_neg_and_pos

    def forward(self, predictions, targets, default_boxes):
        """
        :param predictions: ([batch_size, :, 4], [batch_size, :, 2])
        :param targets: [batch_size, :, 5]
        :param default_boxes: [:, 5]
        :param flag:标注每一个框的是哪个类别的anchor
        :return:
        """
        loc_p, conf_p = predictions
        batch_size = loc_p.size()[0]
        num_default_boxes = default_boxes.size()[0]
        loc_t = torch.Tensor(batch_size, num_default_boxes, 4)
        conf_t = torch.Tensor(batch_size, num_default_boxes)
        # 依次处理batch中的每一个图片
        for batch_idx in range(batch_size):
            ground_truth = targets[batch_idx][:, :-1].data  # [:, 4], [xmin, ymin, xmax, ymax]
            class_label = targets[batch_idx][:, -1].data  # [:]
            default = default_boxes.data   # [:, 4]  [cx, cy, w, h]
            new_match(ground_truth, class_label, default, self.threshold_for_pos, self.threshold_for_neg,
                      loc_t, conf_t, batch_idx)
        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        pos = conf_t == 1.0  # 正样本
        ignore = conf_t == 0.5  # 忽略样本
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_p)
        loc_p = loc_p[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # for hard negative mining
        conf_t = conf_t.long()  # 转化为long类型
        batch_conf = conf_p.view(-1, self.num_classes)
        loss_conf = F.cross_entropy(batch_conf, conf_t.view(-1, 1).squeeze(1), reduce=False)
        loss_conf = loss_conf.unsqueeze(-1)
        # loss_conf = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_conf[pos.view(-1, 1)] = 0   # -float('inf')  0  # 去除掉正样本
        loss_conf[ignore.view(-1, 1)] = 0   #  -float('inf')  0  # 去除掉忽略的样本
        loss_conf = loss_conf.view((batch_size, -1))
        _, loss_conf_idx = loss_conf.sort(dim=1, descending=True)  #
        _, idx_rank = loss_conf_idx.sort(dim=1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.ratio_between_neg_and_pos * num_pos, max=pos.size(1) - 2) + 1  # 防止为0
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # print("num_neg:", torch.sum(neg), "num_pos:", torch.sum(num_pos), "X:", torch.sum((pos.long() + neg.long() == 2)))
        pos_idx = pos.unsqueeze(dim=2).expand_as(conf_p)
        neg_idx = neg.unsqueeze(dim=2).expand_as(conf_p)
        conf_p_all = conf_p[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_conf = F.cross_entropy(conf_p_all, targets_weighted, reduction='sum')

        N = max(num_pos.data.sum().float(), 1)
        loss_loc = loss_loc / N
        loss_conf = loss_conf / N
        return loss_loc, loss_conf






































