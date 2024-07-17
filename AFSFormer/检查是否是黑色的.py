import os
import cv2
import numpy as np

# 指定原始文件夹路径
folder_path = r'C:\Users\Administrator\Desktop\GeoSeg-main\GeoSeg-main\output_our\MANet1'

# 创建保存标记图像的新文件夹
output_folder_path = r'C:\Users\Administrator\Desktop\GeoSeg-main\GeoSeg-main\output_our\MANet1_biaoji'
os.makedirs(output_folder_path, exist_ok=True)

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 遍历图像文件
for image_file in image_files:
    # 构建图像文件的完整路径
    image_path = os.path.join(folder_path, image_file)

    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建一个与图像大小相同的全零矩阵
    mask = np.zeros_like(image)

    # 将像素值为255的像素标记为红色（BGR通道）
    mask[gray_image == 255] = [0, 0, 255]  # 红色标记

    # 将标记后的图像保存到新的文件中
    output_path = os.path.join(output_folder_path, image_file)
    cv2.imwrite(output_path, image + mask)

    print(f"已将像素值为1的像素标记为红色并保存为{output_path}")

print("标记完成")
