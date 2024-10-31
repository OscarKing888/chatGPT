import os
import re

def update_lua_functions(directory):
    # 遍历目录下的所有lua文件
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".lua"):
                file_path = os.path.join(root, file)
                # 处理Lua文件
                modify_lua_file(file_path)

def modify_lua_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    modified_lines = []
    pattern = re.compile(r"(UE\.UTRingBlueprintFunctionLibrary\.GetSoftObjectFullPath\()(.+)(\))")

    for line_num, line in enumerate(lines, start=1):
        # 查找并替换符合要求的行
        match = pattern.search(line)
        if match:
            # 构建Context参数，包含文件名和行号
            context = f'"{os.path.basename(file_path)}:{line_num}"'
            # 替换函数调用，增加Context参数
            new_line = pattern.sub(rf"\1\2, {context}\3", line)
            modified_lines.append(new_line)
        else:
            modified_lines.append(line)

    # 写回修改后的内容
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)
    print(f"Processed file: {file_path}")

# 使用方法：传入需要扫描的目录
directory_path = "./test_lua"
update_lua_functions(directory_path)
