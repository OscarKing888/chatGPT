import unreal
import json

def get_all_assets():
    asset_data_list = []
    
    # 获取资产工具和资产注册器
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    # 搜索所有资产
    asset_data = asset_registry.get_all_assets()
    
    for asset in asset_data:
        asset_name = str(asset.asset_name)
        asset_path = str(asset.package_path)
        asset_class = str(asset.asset_class_path)

        # 构建引用路径，例如 /Script/Engine.Material'/Game/.../AssetName.AssetName'
        #formatted_path =  f"{asset_class}'{asset_path}/{asset_name}.{asset_name}'"
        formatted_path =  asset.get_full_name()
        
        # 创建资产条目并添加到列表
        asset_entry = {
            "Key": asset_name,
            "Path": formatted_path
        }
        asset_data_list.append(asset_entry)
    
    return asset_data_list

def export_to_json(file_path):
    asset_data_list = get_all_assets()
    
    # 导出到JSON文件
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(asset_data_list, json_file, indent=4, ensure_ascii=False)

# 设置导出文件的路径
export_file_path = unreal.Paths.project_saved_dir() + "/AllAssets.json"

# 执行导出
export_to_json(export_file_path)

unreal.log("Assets exported to JSON at: " + export_file_path)
