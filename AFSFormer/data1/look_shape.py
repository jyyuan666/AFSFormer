import numpy as np
from PIL import Image
import os
from os import path as osp
import cv2
from scipy.ndimage import zoom

FLAME_train_imgs = r'D:\data\FLAME2_7\target\JPEGImages'
FLAME_train_masks = r'D:\data\FLAME2_7\target\SegmentationClassPNG'
all_train_imgs = os.listdir(FLAME_train_imgs)
all_train_masks = os.listdir(FLAME_train_masks)


def Crop_generator(all_train_imgs, all_train_masks):

    for img in all_train_imgs:
        img_dir = osp.join(FLAME_train_imgs, img)
        img_np = Image.open(img_dir).convert('RGB').crop((600,20,1624,1044))
        img_np.save(osp.join(r'D:\1\1',img))

    for mask in all_train_masks:
        mask_dir = osp.join(FLAME_train_masks,mask)
        mask_np = Image.open(mask_dir).convert('L').crop((600, 20, 1624, 1044))
        mask_np.save(osp.join(r'D:\1\2',mask))
    return 0

Crop_generator(all_train_imgs,all_train_masks)



'''img = r'D:\yaogan\GeoSeg-main\GeoSeg-main\data\train\Images_1024\image_224.jpg'
mask = r'D:\yaogan\GeoSeg-main\GeoSeg-main\data\train\Masks_1024\image_224.png'
img_pil = Image.open(img).convert('RGB')
mask_pil = Image.open(mask).convert('L')
img_np = np.array(img_pil)
mask_np = np.array(mask_pil)
'''






