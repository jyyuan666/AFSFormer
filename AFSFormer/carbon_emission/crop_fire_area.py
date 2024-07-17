import numpy as np
from PIL import Image
import os
from os import path as osp


results_path = r'D:\yaogan\GeoSeg-main\GeoSeg-main\fig_results\flame2\FireUNET(2)'
save_results_fire_1 = r'D:\yaogan\GeoSeg-main\GeoSeg-main\carbon_emission\results_fire\fire_1'
save_results_fire_2 = r'D:\yaogan\GeoSeg-main\GeoSeg-main\carbon_emission\results_fire\fire_2'
save_results_fire_3 = r'D:\yaogan\GeoSeg-main\GeoSeg-main\carbon_emission\results_fire\fire_3'
save_results_fire_4 = r'D:\yaogan\GeoSeg-main\GeoSeg-main\carbon_emission\results_fire\fire_4'

def crop_all_results(results_path, save_path1, save_path2, save_path3, save_path4):

    all_results = os.listdir(results_path)
    count = 0
    for results in  all_results:
        results_path = osp.join(results_path,results)
        img = Image.open(results_path).convert('L')
        fire_1 = img.crop((70, 60, 210, 200))
        fire_2 = img.crop((180, 150, 320, 290))
        fire_3 = img.crop((467, 56, 567, 156))
        fire_4 = img.crop((880, 610, 930, 660))
        fire_1.save(osp.join(save_path1, str(count) + 'fire_1.png'))
        fire_2.save(osp.join(save_path2, str(count) + 'fire_2.png'))
        fire_3.save(osp.join(save_path3, str(count) + 'fire_3.png'))
        fire_4.save(osp.join(save_path4, str(count) + 'fire_4.png'))
        count += 1
        results_path = r'D:\yaogan\GeoSeg-main\GeoSeg-main\fig_results\flame2\FireUNET(2)'

    print('save successfully')
    return 0

crop_all_results(results_path, save_results_fire_1, save_results_fire_2, save_results_fire_3, save_results_fire_4)