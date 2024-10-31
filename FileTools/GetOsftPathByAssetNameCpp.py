import os
import re

def update_cpp_functions(directory):
    # 遍历目录下的所有cpp和h文件
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp") or file.endswith(".h"):
                file_path = os.path.join(root, file)
                # 修改每个C++文件
                modify_cpp_file(file_path)

def modify_cpp_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    modified_lines = []
    # 匹配 UTRingAssetManager::GetSoftPathByAssetName(参数)
    pattern = re.compile(r"(UTRingAssetManager::GetSoftPathByAssetName\()(.+?)(\))")

    for line in lines:
        # 查找并替换符合要求的行
        match = pattern.search(line)
        if match:
            # 使用 GET_CONTEXT_INFO() 宏替换 Context 参数
            new_line = pattern.sub(rf"\1\2, TR_ASSET_CONTEXT\3", line)
            modified_lines.append(new_line)
        else:
            modified_lines.append(line)

    # 写回修改后的内容
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)
    print(f"Processed file: {file_path}")


# 使用方法：传入需要扫描的目录
directory_path = "./test_lua"
update_cpp_functions(directory_path)
