import os
from PIL import Image

# 图片文件夹路径
folder_path = r'C:\Users\Administrator\Desktop\test'

# 获取文件夹中所有图片文件的路径
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]

# 读取第一张图片的属性
first_image = Image.open(image_paths[0])
first_image_width, first_image_height = first_image.size
first_image_mode = first_image.mode

# 按照顺序重命名图片并保持属性相同
for i, image_path in enumerate(image_paths):
    # 构建新的文件名
    new_filename = os.path.join(folder_path, f"{i+1}.jpg")
    # 读取原始图片
    image = Image.open(image_path)
    # 保持图片属性与第一张图片相同
    image = image.convert(first_image_mode)
    image = image.resize((first_image_width, first_image_height))
    # 保存图片
    image.save(new_filename)
    print(f"重命名 {image_path} 为 {new_filename}")
