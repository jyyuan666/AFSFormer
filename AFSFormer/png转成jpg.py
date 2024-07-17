from PIL import Image
import os

# 文件夹路径
folder_path = r"C:\Users\Administrator\Desktop\test\Images_1024"

# 获取文件夹中的所有PNG图片文件
png_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.png')]

# 将PNG图片转换为JPEG格式
for png_file in png_files:
    # 构建文件的完整路径
    file_path = os.path.join(folder_path, png_file)
    # 打开PNG图片
    image = Image.open(file_path)
    # 构建新的文件名
    new_file_name = os.path.splitext(png_file)[0] + ".jpg"
    # 构建新的文件路径
    new_file_path = os.path.join(folder_path, new_file_name)
    # 转换为JPEG格式并保存
    image.convert("RGB").save(new_file_path, "JPEG")
    # 删除原始PNG图片
    os.remove(file_path)

print("完成将PNG图片转换为JPEG格式！")
