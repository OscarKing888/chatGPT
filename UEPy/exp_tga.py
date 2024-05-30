import unreal
import os

# 设置输入和输出路径
input_content_directory = "/Game/FPWeapon/"
output_directory = "C:/ExportedTextures"

# 获取Asset工具
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

# 获取所有Texture类型的资产
asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
texture_assets = asset_registry.get_assets_by_path(input_content_directory, recursive=True, include_only_on_disk_assets=True)

last_export = ""

# 定义一个导出函数
def export_texture_to_tga(texture_asset, output_path):
    # 获取资产数据
    texture = texture_asset.get_asset() # unreal.EditorAssetLibrary.load_asset(texture_asset.package_path)

    if not texture:
        print(f"Failed to load asset {texture_asset.package_path}")
        return False

    # 创建导出路径
    
    pkg_path =  str(texture_asset.package_name)
    #relative_path = texture_asset.package_path.replace(input_content_directory, "").lstrip("/")
    relative_path = pkg_path.replace(input_content_directory, "").lstrip("/")
    export_directory = os.path.join(output_path, os.path.dirname(relative_path))

    print(f"Exporting {pkg_path}  {export_directory} {relative_path}")

    if not os.path.exists(export_directory):
        os.makedirs(export_directory)

    # 设置导出文件名
    export_file_path = os.path.join(export_directory, relative_path + ".tga")
    last_export = export_file_path

    # 创建导出任务
    export_task = unreal.AssetExportTask()
    export_task.object = texture
    export_task.filename = export_file_path
    export_task.selected = False
    export_task.automated = True
    export_task.replace_identical = True
    export_task.prompt = False
    success = unreal.Exporter.run_asset_export_task(export_task)

    return success

# 遍历所有找到的贴图并导出
for texture_asset in texture_assets:
    export_success = export_texture_to_tga(texture_asset, output_directory)
    if export_success:
        print(f"Successfully exported {last_export} to {output_directory}")
    else:
        print(f"Failed to export {last_export}")

print("Export process completed.")
