import os

# 文件夹路径
folder_path = r"C:\Users\Administrator\Desktop\test\label"

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 遍历每个文件
for file_name in files:
    # 检查文件名是否包含"_json_label"部分
    if "_json_label" in file_name:
        # 构建新的文件名
        new_file_name = file_name.replace("_json_label", "")
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        # 重命名文件
        os.rename(file_path, new_file_path)

print("完成重命名图片文件！")
