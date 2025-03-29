import os
import json
import argparse


def add_plugin_to_uproject(uproject_path, plugin_names):
    """
    向指定的.uproject文件中添加插件（如果不存在的话）
    :param uproject_path:.uproject文件的路径
    :param plugin_names: 要添加的插件名称列表
    """
    with open(uproject_path, 'r', encoding='utf-8') as f:
        content = json.load(f)

    existing_plugins = [plugin["Name"] for plugin in content.get("Plugins", [])]
    for plugin_name in plugin_names:
        if plugin_name not in existing_plugins:
            new_plugin = {
                "Name": plugin_name,
                "Enabled": True
            }
            content.setdefault("Plugins", []).append(new_plugin)

    with open(uproject_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=4)


def process_directory(directory, plugin_names):
    """
    递归处理指定目录下的所有.uproject文件
    :param directory: 要处理的目录路径
    :param plugin_names: 要添加的插件名称列表
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.uproject'):
                uproject_file_path = os.path.join(root, file)
                add_plugin_to_uproject(uproject_file_path, plugin_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a directory to add plugins to.uproject files.')
    parser.add_argument('directory', type=str, help='The directory path to process')
    parser.add_argument('plugins', nargs='+', type=str, help='The list of plugin names to add')
    args = parser.parse_args()

    process_directory(args.directory, args.plugins)