import os
import chardet
from collections import defaultdict

def rename_files_and_replace_content(directory):
    # 指定要处理的文件扩展名
    file_extensions = ['.h', '.cpp', '.cs']

    # 收集旧文件名到新文件名的映射
    old_to_new_filenames = {}
    files_to_process = []

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 获取文件的扩展名
            _, ext = os.path.splitext(filename)
            if ext not in file_extensions:
                continue  # 跳过不在指定扩展名列表中的文件

            old_path = os.path.join(root, filename)

            # 生成新的文件名
            if filename.startswith('RPG'):
                new_filename = 'TPS' + filename[2:]
            else:
                new_filename = 'TPS' + filename

            new_path = os.path.join(root, new_filename)
            old_to_new_filenames[old_path] = new_path
            files_to_process.append((old_path, new_path))

    # 检查新文件名是否有冲突
    new_paths = [new_path for old_path, new_path in files_to_process]
    if len(new_paths) != len(set(new_paths)):
        print("检测到新文件名冲突，无法继续。请先解决冲突。")
        return

    # 重命名文件
    for old_path, new_path in files_to_process:
        # 创建可能不存在的父目录
        new_dir = os.path.dirname(new_path)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        # 处理可能的文件名冲突
        if os.path.exists(new_path):
            print(f"文件 {new_path} 已存在，跳过重命名 {old_path}。")
            continue

        os.rename(old_path, new_path)
        print(f"已将 {old_path} 重命名为 {new_path}")

    # 按文件名长度从长到短排序，避免部分匹配问题
    old_filenames_sorted = sorted(old_to_new_filenames.keys(), key=lambda x: len(os.path.basename(x)), reverse=True)

    # 替换文件内容中的文件名
    for new_path in old_to_new_filenames.values():
        if os.path.isdir(new_path):
            continue

        # 检测文件编码
        with open(new_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            if not encoding:
                print(f"无法检测文件 {new_path} 的编码，跳过该文件。")
                continue

        # 将文件内容解码为 Unicode
        try:
            content = raw_data.decode(encoding)
        except UnicodeDecodeError:
            print(f"解码文件 {new_path} 时出现错误，跳过该文件。")
            continue

        # 替换旧文件名为新文件名（大小写精确匹配）
        for old_path in old_filenames_sorted:
            old_filename = os.path.basename(old_path)
            new_filename = os.path.basename(old_to_new_filenames[old_path])
            content = content.replace(old_filename, new_filename)

        # 将内容编码为 UTF-8 并写回文件
        try:
            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"已替换 {new_path} 中的内容，并转换为 UTF-8 编码。")
        except UnicodeEncodeError:
            print(f"写入文件 {new_path} 时出现编码错误，跳过该文件。")
            continue

if __name__ == "__main__":
    directory = '.'  # 替换为你的目录路径
    rename_files_and_replace_content(directory)
