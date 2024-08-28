import unreal
import pandas as pd
import os
import sys

# 获取所有属性类型
#unreal_attributes = [attr for attr in dir(unreal) if attr.endswith("Property")]
#unreal_attributes = dir(unreal)
# 输出所有属性类型
#for attr in unreal_attributes:
#    print(attr)

# 读取Excel表格
excel_path = "E:/chatGPT/UEPy/Test.xlsx"  # 修改为你的Excel文件路径

def export_excel(excel_path):
    # 读取Excel表格
    df = pd.read_excel(excel_path)

    # 打印整个DataFrame以调试
    print(df)

    # 检查DataFrame行数
    if len(df) < 3:
        raise ValueError("Excel文件中至少应有三行：变量名、类型和中文描述")

    # 提取文件名并处理为Struct名称和路径
    excel_filename = os.path.basename(excel_path)  # 获取文件名（带扩展名）
    file_name_without_ext = os.path.splitext(excel_filename)[0]  # 获取文件名（不带扩展名）

    # 根据文件名生成struct_name和package_path
    struct_name = f"F{file_name_without_ext}Row"  # 生成Struct名称，例如FItemDataRow
    package_path = "/Game/DataTable/Struct"  # 生成路径，例如/Game/DataTable/ItemData

    # 提取表格数据
    variable_names = df.iloc[0].tolist()  # 第一行应该是变量名
    variable_types = df.iloc[1].tolist()  # 第二行应该是类型
    variable_chinese_names = df.iloc[2].tolist() if len(df) > 2 else [""] * len(variable_names)  # 第三行中文描述（可选）

    # 创建一个新的结构体
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    structure_factory = unreal.StructureFactory()

    struct_asset_path = f"{package_path}/{struct_name}"

    if unreal.EditorAssetLibrary.does_asset_exist(struct_asset_path):
        unreal.EditorAssetLibrary.delete_asset(struct_asset_path)
        unreal.log(f"Deleted existing asset at {struct_asset_path}")

    #if not unreal.EditorAssetLibrary.does_asset_exist(struct_asset_path):
    # 资产不存在，创建新的资产
    struct_asset = asset_tools.create_asset(struct_name, package_path, None, structure_factory)
        
    struct_class = unreal.EditorAssetLibrary.load_asset(struct_asset_path)

    # 为结构体添加成员变量
    for name, var_type, chinese_name in zip(variable_names, variable_types, variable_chinese_names):
        # 使用unreal.StructureFactory添加新变量
        variable_type = var_type#getattr(unreal, var_type)    
        if not unreal.EditorExBPLibrary.has_variable_in_struct(struct_class, name):
            print(f"====== Adding new var:{variable_type} -  {name} - {chinese_name}")
            unreal.EditorExBPLibrary.add_variable_to_struct(struct_class, name, variable_type, chinese_name)


    # 删除MemberVar_0这个变量，初始创建的结构体会有这个变量
    if unreal.EditorExBPLibrary.has_variable_in_struct(struct_class, "MemberVar_0"):
        unreal.EditorExBPLibrary.remove_variable_from_struct(struct_class, "MemberVar_0")

    # 保存结构体
    unreal.EditorAssetLibrary.save_asset(struct_asset_path)



    package_path = "/Game/DataTable"
    # 创建Data Table
    data_table_name = f"{file_name_without_ext}DataTable"  # 生成Data Table名称，例如ItemDataDataTable
    data_table_path = f"{package_path}/{data_table_name}"

    data_table_factory = unreal.DataTableFactory()
    data_table_factory.struct = struct_class

    # 在创建新资产之前删除已有的资产
    if unreal.EditorAssetLibrary.does_asset_exist(data_table_path):
        unreal.EditorAssetLibrary.delete_asset(data_table_path)
        unreal.log(f"Deleted existing asset at {data_table_path}")

    data_table_asset = asset_tools.create_asset(data_table_name, package_path, None, data_table_factory)

    # 保存Data Table
    unreal.EditorAssetLibrary.save_asset(data_table_path)


if __name__ == "__main__":
    
    print(sys.argv)

    if len(sys.argv) > 1:
        excel_path = sys.argv[1]

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel文件不存在：{excel_path}")
    
    print(f"==========Exporting Excel to Unreal: {excel_path}")
    export_excel(excel_path)