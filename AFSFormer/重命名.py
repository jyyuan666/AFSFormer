import os

# 文件夹路径
folder_path = r"C:\Users\Administrator\Desktop\test\Masks"

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]

# 排序图片文件
image_files.sort()

# 重新命名图片文件
for i, image_file in enumerate(image_files, start=73):
    # 构建新的文件名
    new_file_name = f"fire_{i}.png"
    # 构建文件的完整路径
    file_path = os.path.join(folder_path, image_file)
    new_file_path = os.path.join(folder_path, new_file_name)
    # 重命名文件
    os.rename(file_path, new_file_path)

print("完成重新命名图片文件！")
