import os

# 指定要修改的文件夹路径
folder_path = "/Users/momochan/chapter0/machikomomono_github/machikomomo.github.io/content/img_cloud"

# 获取文件夹中所有文件的文件名
file_names = os.listdir(folder_path)

# 循环遍历每个文件名，将包含 "xx" 的字符串替换为空字符串
for file_name in file_names:
    new_file_name = file_name.replace("截屏", "")
    new_file_name = new_file_name.replace("下午", "")
    new_file_name = new_file_name.replace(" ", "")
    # 使用 os.rename() 函数重命名文件
    os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
