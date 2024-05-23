import os
import re
from graphviz import Digraph

# 读取目录及其所有子目录中的所有.h文件
def read_files_from_directory(directory):
    file_contents = {}
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".h"):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_contents[file_path] = file.read()
    return file_contents

# 提取以 "class I" 开头的类
def extract_classes(content):
    class_pattern = re.compile(r'\bclass\s+(\w+)')
    #class_pattern = re.compile(r'\bclass\s+(I\w+)')
    classes = class_pattern.findall(content)
    return classes

# 提取类中的函数引用
def extract_references(content, class_name):
    references = set()
    # 提取类定义部分
    class_pattern = re.compile(r'\bclass\s+' + re.escape(class_name) + r'\b.*?{(.*?)};', re.DOTALL)
    class_body = class_pattern.search(content)
    if class_body:
        class_body_content = class_body.group(1)
        # 查找函数中的引用
        ref_pattern = re.compile(r'\b(I\w+)\b')
        potential_refs = ref_pattern.findall(class_body_content)
        for ref in potential_refs:
            if ref != class_name:  # 避免自引用
                references.add(ref)
    return references

# 提取类的继承关系
def extract_inheritance(content, class_name):
    inheritance_pattern = re.compile(r'\bclass\s+' + re.escape(class_name) + r'\s*:\s*public\s+(I\w+)')
    inheritance = inheritance_pattern.findall(content)
    return inheritance

# 提取所有类对特定类的引用关系
def extract_all_references(file_contents, target_classes):
    all_references = {}
    inheritance_map = {}
    
    for file_name, content in file_contents.items():
        classes = extract_classes(content)
        for target_class in target_classes:
            if target_class in classes:
                references = extract_references(content, target_class)
                inheritance = extract_inheritance(content, target_class)
                all_references[target_class] = references
                inheritance_map[target_class] = inheritance

        for cls in classes:
            references = extract_references(content, cls)
            for target_class in target_classes:
                if target_class in references:
                    if cls in all_references:
                        all_references[cls].add(target_class)
                    else:
                        all_references[cls] = {target_class}
            inheritance = extract_inheritance(content, cls)
            if inheritance:
                inheritance_map[cls] = inheritance
                
    return all_references, inheritance_map


# 生成UML图
def generate_uml(classes_interfaces, inheritance_map, target_classes, output_path):
    dot = Digraph(comment='Classes and Interfaces UML Diagram')

    for cls, refs in classes_interfaces.items():
        # 设置 target_classes 中类的颜色
        if cls in target_classes:
            dot.node(cls, cls, shape='box', style='filled', fillcolor='lightblue')
        else:
            dot.node(cls, cls, shape='box')
        for ref in refs:
            if ref in target_classes:
                dot.node(ref, ref, shape='box', style='filled', fillcolor='lightblue')
            else:
                dot.node(ref, ref, shape='box')
            dot.edge(cls, ref)
    
    # 添加继承关系
    for cls, bases in inheritance_map.items():
        for base in bases:
            dot.edge(base, cls, arrowhead='onormal')

    # 将输出格式设置为 PNG
    dot.format = 'png'
    dot.render(output_path, view=True)

# 主程序
def main(directory, output_path, target_classes):
    file_contents = read_files_from_directory(directory)
    classes_interfaces, inheritance_map = extract_all_references(file_contents, target_classes)
    generate_uml(classes_interfaces, inheritance_map, target_classes, output_path)

# 指定目录和输出路径
target_classes = ["IDetailLayoutBuilder", "IPropertyTypeCustomization", "IDetailCustomization"]  # 替换为你要收集引用关系的类名列表


directory = "./"  # 替换为你实际的目录路径
output_path = "./uml"

# 运行主程序
main(directory, output_path, target_classes)
