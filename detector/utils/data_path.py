import os
import numpy as np
import pandas as pd
import scipy.io as scio


wider_face_image_path = '/root/data/faceboxes/wider_face'
wider_face_label_path = '/root/private/Face/data_label/wider_face/wider_face_split'

afw_image_path = '/root/data/faceboxes/afw_images'

pascal_image_path = '/root/data/faceboxes/pascal_images'


# for wider_face
# image_name = []
# gt_label = []
# state = 'val'
# if state == 'train':
#     image_path = os.path.join(wider_face_image_path, 'WIDER_train', 'images')
#     image_path2 = os.path.join(wider_face_label_path, 'wider_face_train_bbx_gt.txt')
#     gt_label_path = os.path.join(wider_face_label_path, 'wider_face_train.mat')
#     file = pd.read_table(image_path2, header=None)
#     for idx in file.index:
#         name = file.loc[idx].values[0]
#         if name[-1] == 'g':
#             image_name.append(os.path.join(image_path, name))
#     print(len(image_name))
#     gt = scio.loadmat(gt_label_path)['face_bbx_list']
#     for i in range(len(gt)):
#         for j in range(len(gt[i][0])):
#             gt_label.append(gt[i][0][j][0])
#     print(len(gt_label))
#     np.save('train_images.npy', image_name)
#     np.save('train_label.npy', gt_label)
#
# if state == 'val':
#     image_path = os.path.join(wider_face_image_path, 'WIDER_val', 'images')
#     image_path2 = os.path.join(wider_face_label_path, 'wider_face_val_bbx_gt.txt')
#     gt_label_path = os.path.join(wider_face_label_path, 'wider_face_val.mat')
#     file = pd.read_table(image_path2, header=None)
#     for idx in file.index:
#         name = file.loc[idx].values[0]
#         if name[-1] == 'g':
#             image_name.append(os.path.join(image_path, name))
#     gt = scio.loadmat(gt_label_path)['face_bbx_list']
#     for i in range(len(gt)):
#         for j in range(len(gt[i][0])):
#             gt_label.append(gt[i][0][j][0])
#     np.save('val_images.npy', image_name)
#     np.save('val_label.npy', gt_label)
#     # val = pd.DataFrame({'image': image_name, 'gt_label': gt_label})
#     # val.to_csv(os.path.join(os.getcwd(), 'val.csv'), index=False)
#
# if state == 'test':
#     image_path = os.path.join(wider_face_image_path, 'WIDER_test', 'images')
#     image_path2 = os.path.join(wider_face_label_path, 'wider_face_test_filelist.txt')
#     file = pd.read_table(image_path2, header=None)
#     for idx in file.index:
#         name = file.loc[idx].values[0]
#         if name[-1] == 'g':
#             image_name.append(os.path.join(image_path, name))
#     np.save('test_images.npy', image_name)

# for afw
# image_name = []
# for name in os.listdir(afw_image_path):
#     if name[-3:] == 'jpg':
#         image_name.append(os.path.join(afw_image_path, name))
# print(len(image_name))
# np.save('afw.npy', image_name)

# for pascal
image_name = []
for name in os.listdir(pascal_image_path):
    if name[-3:] == 'jpg':
        image_name.append(os.path.join(pascal_image_path, name))
print(len(image_name))
np.save('pascal.npy', image_name)
