import os
import shutil

# 原始文件夹路径
source_folder = r"C:\Users\Administrator\Desktop\test\mask"

# 新文件夹路径
target_folder = r"C:\Users\Administrator\Desktop\test\label"

# 获取原始文件夹下的所有文件夹
subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

# 遍历每个文件夹
for folder in subfolders:
    # 获取文件夹的名字
    folder_name = os.path.basename(folder)
    # 构建label.png的路径
    label_path = os.path.join(folder, "label.png")
    # 构建目标保存路径
    target_path = os.path.join(target_folder, f"{folder_name}_label.png")
    # 复制文件到目标路径
    shutil.copy(label_path, target_path)

print("完成保存label.png图片到新文件夹！")
