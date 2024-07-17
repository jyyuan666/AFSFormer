import os
from PIL import Image

# 文件夹路径
folder_path = r"C:\Users\Administrator\Desktop\test\Masks"

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 调整图片尺寸为1024x1024
for image_file in image_files:
    # 构建文件的完整路径
    file_path = os.path.join(folder_path, image_file)
    # 打开图片
    image = Image.open(file_path)
    # 调整尺寸为1024x1024
    resized_image = image.resize((1024, 1024))
    # 保存调整尺寸后的图片
    resized_image.save(file_path)

print("完成将图片调整为1024x1024尺寸！")
