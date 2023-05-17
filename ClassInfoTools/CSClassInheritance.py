import os, sys
import re
from collections import defaultdict
import argparse

# 建立一个字典来保存类和接口的继承关系
inheritance = defaultdict(list)

def parse_directory(directory):
    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cs'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    # 使用正则表达式查找类和接口的定义
                    # 修改后的正则表达式可以匹配一个类或接口继承自多个基类或接口的情况
                    matches = re.findall(r'(class|interface)\s+(\w+)\s*:\s*([\w\,\s]+)', content)
                    for match in matches:
                        derived = match[1]
                        # 分割基类或接口列表
                        bases = [base.strip() for base in match[2].split(',')]
                        for base in bases:
                            # 保存继承关系
                            inheritance[base].append(derived)


# 函数用于生成继承结构图
def print_inheritance(base, prefix=''):
    print(prefix + base)
    if base in inheritance:
        for i, derived in enumerate(inheritance[base]):
            if i == len(inheritance[base]) - 1:
                print_inheritance(derived, prefix + '└── ')
            else:
                print_inheritance(derived, prefix + '├── ')

def main():
    # 创建命令行解析器
    # parser = argparse.ArgumentParser(description='Parse C# files in a directory and output inheritance structure.')
    # parser.add_argument('directory', type=str, help='The directory to parse')
    
    # # 解析命令行参数
    # args = parser.parse_args()
    
    parse_directory(sys.argv[1])
    
    # 打印继承结构图
    for base in inheritance:
        if base not in sum(inheritance.values(), []):
            print_inheritance(base)

if __name__ == '__main__':
    main()
