import os
from PIL import Image

folder_path = r"C:\Users\Administrator\Desktop\GeoSeg-main\GeoSeg-main\output_our\unet1"
output_folder_path = r"C:\Users\Administrator\Desktop\GeoSeg-main\GeoSeg-main\output_our\unet1_red"

# 创建保存修改后图像的文件夹
os.makedirs(output_folder_path, exist_ok=True)

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")]

# 遍历每个图像文件
for image_file in image_files:
    # 打开图像文件
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)

    # 将蓝色区域变为红色
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        # 判断像素是否为蓝色
        if item[0] < 100 and item[1] < 100 and item[2] > 150:
            # 将蓝色像素变为红色
            new_data.append((255, 0, 0, item[3]))
        else:
            new_data.append(item)
    image.putdata(new_data)

    # 保存修改后的图像到新的文件夹，使用原始图片的名字
    output_path = os.path.join(output_folder_path, image_file)
    image.save(output_path)
