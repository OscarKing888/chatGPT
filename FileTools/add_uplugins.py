import os
import json
import argparse

# 配置变量
new_plugins = ["GameplayAbilities"]  # 要添加的插件列表

def add_plugins_to_uproject(file_path, plugins_to_add):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 确保 "Plugins" 键存在
        if "Plugins" not in data:
            data["Plugins"] = []

        # 获取当前已存在的插件名
        existing_plugins = {plugin['Name'] for plugin in data["Plugins"]}

        # 添加新插件
        for plugin_name in plugins_to_add:
            if plugin_name not in existing_plugins:
                data["Plugins"].append({
                    "Name": plugin_name,
                    "Enabled": True
                })

        # 将更新后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"Updated: {file_path}")

    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def process_directory(directory, plugins_to_add):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.uproject'):
                file_path = os.path.join(root, file)
                add_plugins_to_uproject(file_path, plugins_to_add)

def main():
    parser = argparse.ArgumentParser(description="Add plugins to .uproject files.")
    parser.add_argument("directory", help="The directory to process.")
    args = parser.parse_args()

    process_directory(args.directory, new_plugins)

if __name__ == "__main__":
    main()
