import json
import sys
import os

def load_json_file(json_file_path):
    """加载 JSON 文件并返回解析后的数据"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_keys_from_txt(txt_file_path):
    """从 TXT 文件中读取 Key 列表"""
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def find_paths(keys, assets_data):
    """根据 Key 查找对应的 Path"""
    path_dict = {asset["Key"]: asset["Path"] for asset in assets_data}
    return [path_dict.get(key, "Path not found") for key in keys]

def save_output(output_file_path, paths):
    """保存输出到 TXT 文件"""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for path in paths:
            f.write(path + '\n')

def main(json_file_path, txt_file_path, output_file_path=None):
    # 加载 JSON 数据和 Key 列表
    assets_data = load_json_file(json_file_path)
    keys = load_keys_from_txt(txt_file_path)

    # 查找对应的 Path
    paths = find_paths(keys, assets_data)

    # 确定输出文件路径
    if not output_file_path:
        # 添加 '_out' 后缀到输入 txt 文件的文件名
        output_file_path = f"{os.path.splitext(txt_file_path)[0]}_out.txt"

    # 保存输出结果
    save_output(output_file_path, paths)
    print(f"Output saved to {output_file_path}")

if __name__ == "__main__":
    # 获取命令行参数
    if len(sys.argv) < 2:
        print("Usage: python script.py <AllAssets.json> <input.txt> [output.txt]")
    else:
        json_file_path = sys.argv[1]
        txt_file_path = sys.argv[2]
        output_file_path = sys.argv[3] if len(sys.argv) > 3 else None
        main(json_file_path, txt_file_path, output_file_path)
