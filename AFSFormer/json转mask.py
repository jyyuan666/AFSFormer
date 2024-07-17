import os
import cv2
import json

# JSON文件夹路径
folder_path = r'C:\Users\Administrator\Desktop\test\mask'

# 遍历文件夹中的所有JSON文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        # 构建JSON文件的完整路径
        json_file_path = os.path.join(folder_path, file_name)

        # 读取JSON文件
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)

        # 获取JSON数据中的蒙版信息
        mask_data = json_data['mask']

        # 创建空白蒙版图像
        mask_image = np.zeros((mask_data['height'], mask_data['width']), dtype=np.uint8)

        # 遍历蒙版数据，将对应位置的像素值设置为1
        for segment in mask_data['segments']:
            for point in segment['points']:
                x, y = int(point['x']), int(point['y'])
                mask_image[y, x] = 1

        # 保存蒙版图像
        mask_image_path = os.path.join(folder_path, f"{file_name.split('.')[0]}.png")
        cv2.imwrite(mask_image_path, mask_image)
        print(f"已生成蒙版图像：{mask_image_path}")
