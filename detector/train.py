import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from model.FaceBoxes import FaceBoxes
from model.HRFaceBoxes import HRFaceBoxes
from utils.data_load import DataSet, AnnotationTransform, ToTensor, detection_collate
from utils.data_augment import DataAug
from utils.default_boxes import DefaultBox, HRDefaultBox
from loss_and_metrics.loss_and_metrics import NewMultiBoxesLoss, MultiBoxesLoss
import config


GPU = [0]


def adjust_lr(optimizer, epoch):
    if epoch < 10:
        lr = 1e-3
    elif epoch < 15:
        lr = 8e-4
    elif epoch < 20:
        lr = 6e-4
    elif epoch < 30:
        lr = 4e-4
    elif epoch < 40:
        lr = 2e-4
    elif epoch < 50:
        lr = 1e-4
    elif epoch < 70:
        lr = 8e-5
    elif epoch < 90:
        lr = 6e-5
    elif epoch < 120:
        lr = 5e-5
    elif epoch < 140:
        lr = 2e-5
    elif epoch < 150:
        lr = 1e-5
    else:
        lr = 5e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(model, criterion, data_loader, optimizer, default_boxes, config, epoch, train_logs):
    model.train()
    total_loss = 0
    total_loc_loss = 0
    total_conf_loss = 0
    data_process = tqdm(data_loader)  # tqdm is a tool to display information when train or test
    for batch_data in data_process:
        image, label = batch_data['image'], batch_data['label']
        if torch.cuda.is_available():
            image, label = image.cuda(device=GPU[0]), [anno.cuda(device=GPU[0]) for anno in label]
        optimizer.zero_grad()
        out = model(image)
        loss_loc, loss_conf = criterion(out, label, default_boxes)
        loss = loss_loc * config.loc_weights * 2 + loss_conf * config.conf_weights
        total_loss += loss.item()
        total_loc_loss += loss_loc.item()
        total_conf_loss += loss_conf.item()
        loss.backward()
        optimizer.step()
        data_process.set_description_str("epoch:{}".format(epoch))
        data_process.set_postfix({"loss_loc": "{:.6f}".format(loss_loc.item()),
                                  "loss_conf": "{:.6f}".format(loss_conf.item()),
                                  "loss": "{:.6f}".format(loss.item())})
    train_logs.write("Epoch:{}, loss_loc is {:.6f},"
                     " loss_conf is {:.6f},loss is {:.6f} \n".format(epoch, total_loc_loss / len(data_loader),
                                                                     total_conf_loss / len(data_loader),
                                                                     total_loss / len(data_loader)))
    train_logs.flush()


def train(model_name, model_path, start_epoch):
    cfg = config
    os.makedirs(cfg.save_path, exist_ok=True)
    if model_name == 'OriginalFaceBoxes':
        model = FaceBoxes(phase='train', num_classes=cfg.num_classes)
        d_box = DefaultBox(img_size=(cfg.img_dim2, cfg.img_dim2))
        img_dim = cfg.img_dim2
        criterion = MultiBoxesLoss(num_classes=config.num_classes, threshold_for_match=0.35,
                                   threshold_for_hard_gt=0.2, ratio_between_neg_and_pos=3)
    elif model_name == 'HRFaceBoxes':
        model = HRFaceBoxes(phase='train', num_classes=2)
        d_box = HRDefaultBox(img_size=(cfg.img_dim2, cfg.img_dim2))
        img_dim = cfg.img_dim2
        criterion = NewMultiBoxesLoss(num_classes=config.num_classes, threshold_for_pos=cfg.threshold_for_pos,
                                      threshold_for_neg=cfg.threshold_for_neg,
                                      ratio_between_neg_and_pos=cfg.ratio_between_neg_and_pos)

    cudnn.benchmark = True
    if torch.cuda.is_available():
        model = model.cuda(device=GPU[0])
        model = torch.nn.DataParallel(model, device_ids=GPU)

    if model_path is not None:
        model_dict = torch.load(os.path.join(cfg.save_path, model_path))
        model.load_state_dict(model_dict)

    with torch.no_grad():
        default_boxes = d_box.forward().cuda(device=GPU[0])

    kwargs = {'num_workers': 5, 'pin_memory': False} if torch.cuda.is_available() else {}
    train_dataset = DataSet(transform=transforms.Compose([AnnotationTransform(),
                                                          DataAug(img_dim, cfg.rgb_mean), ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=48*len(GPU), shuffle=True, drop_last=True,
                                  collate_fn=detection_collate, **kwargs)

    train_logs = open(os.path.join(cfg.save_path, model_name + '.csv'), 'w')

    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(start_epoch, cfg.EPOCHS):
        adjust_lr(optimizer, epoch)
        train_epoch(model, criterion, train_dataloader, optimizer, default_boxes, cfg, epoch, train_logs)
        if epoch < 50:
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(cfg.save_path, model_name + '_{}.pth'.format(epoch)))
        elif epoch < 70:
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(cfg.save_path, model_name + '_{}.pth'.format(epoch)))
        else:
            if epoch % 3 == 0:
                torch.save(model.state_dict(), os.path.join(cfg.save_path, model_name + '_{}.pth'.format(epoch)))


if __name__ == '__main__':
    # train('FaceBoxes', None, 0)
    train('HRFaceBoxes', None, 0)










