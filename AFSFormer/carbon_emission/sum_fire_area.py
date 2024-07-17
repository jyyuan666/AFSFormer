import numpy as np
from PIL import Image
import os
from os import path as osp


paths = [r'D:\yaogan\GeoSeg-main\GeoSeg-main\carbon_emission\results_fire\fire_1',
        r'D:\yaogan\GeoSeg-main\GeoSeg-main\carbon_emission\results_fire\fire_2',
        r'D:\yaogan\GeoSeg-main\GeoSeg-main\carbon_emission\results_fire\fire_3',
        r'D:\yaogan\GeoSeg-main\GeoSeg-main\carbon_emission\results_fire\fire_4']

def sum_fire_area(paths):

    for path in paths:
        all_results = os.listdir(path)

        for results in all_results:
            img = Image.open(osp.join(path, results)).convert('L')
            img_np = np.array(img, dtype=np.uint8)
            H, W = img_np.shape[0], img_np.shape[1]
            all_area = H * W
            fire_area = np.sum(img_np == 255)
            ratio = fire_area / all_area

            print('fire_area_name:', results.split('.')[0], 'ratio:', ratio)


    print('finish!')

    return 0

sum_fire_area(paths)
