import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        return torch.from_numpy(image.copy()), label.copy()


class ToTensor1(object):
    def __call__(self, sample):
        image, label, semantic = sample
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        return torch.from_numpy(image.copy()), label.copy(), torch.from_numpy(semantic.copy())



class AnnotationTransform(object):
    def __call__(self, sample):
        image, targets = sample
        targets_t = []
        for i in range(len(targets)):
            # xmin, ymin, w, h to xmin, ymin, xmax, ymax
            box = np.asarray(targets[i], dtype=np.float)
            box[2:] = box[:2] + box[2:]
            box = np.hstack([box, 1])
            targets_t.append(box)
        return image, np.asarray(targets_t)


class DataSet(Dataset):
    def __init__(self, transform=None):
        super(DataSet, self).__init__()
        self.train_images = np.load(os.path.join(os.getcwd(), 'utils', 'data_list', 'train_images.npy'), allow_pickle=True)
        self.train_label = np.load(os.path.join(os.getcwd(), 'utils', 'data_list', 'train_label.npy'), allow_pickle=True)
        self.transfrom = transform

    def __len__(self):
        return len(self.train_label)

    def __getitem__(self, idx):
        image = cv2.imread(self.train_images[idx])
        label = self.train_label[idx]
        sample = (image, label)
        if self.transfrom:
            sample = self.transfrom(sample)
        return sample


class ValDataSet(Dataset):
    def __init__(self):
        super(ValDataSet, self).__init__()
        self.val_images = np.load(os.path.join(os.getcwd(), 'utils', 'data_list', 'val_images.npy'), allow_pickle=True)
        self.val_label = np.load(os.path.join(os.getcwd(), 'utils', 'data_list', 'val_label.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.val_label)

    def __getitem__(self, idx):
        image = cv2.imread(self.val_images[idx])
        # image = np.float32(image)
        height, width, _ = image.shape
        label = self.val_label[idx]
        image = image - (104, 117, 123)
        image = np.transpose(image, (2, 0, 1))
        # image = torch.from_numpy(image).float()
        targets = []
        for i in range(len(label)):
            # x, y, w, h to xmin, ymin, xmax, ymax
            box = np.asarray(label[i], dtype=np.float)
            box[2:] = box[:2] + box[2:]
            box[0::2] = box[0::2] / width
            box[1::2] = box[1::2] / height
            box = np.hstack([box, 1])
            targets.append(box)

        return {"image": torch.from_numpy(image.copy()).float(), "label": torch.from_numpy(np.asarray(targets)).float(),
                "HW": (height, width)}


def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
    return {'image': torch.stack(imgs, 0), 'label': targets}


def detection_collate1(batch):
    imgs = []
    targets = []
    semantic = []
    for _, sample in enumerate(batch):
        for i, tup in enumerate(sample):
            if i == 0:
                imgs.append(tup)
            elif i == 1:
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
            elif i == 2:
                semantic.append(tup)

    return {'image': torch.stack(imgs, 0), 'label': targets, 'semantic_label': torch.stack(semantic, 0)}







