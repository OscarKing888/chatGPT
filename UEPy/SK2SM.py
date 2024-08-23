import unreal

def extract_and_save_static_meshes_from_skeletal_mesh(skeletal_mesh_path):
    # Load the skeletal mesh asset
    skeletal_mesh = unreal.EditorAssetLibrary.load_asset(skeletal_mesh_path)
    if not skeletal_mesh:
        unreal.log_error(f"Could not load Skeletal Mesh at path: {skeletal_mesh_path}")
        return

    # Get the directory of the skeletal mesh
    skeletal_mesh_directory = unreal.Paths.get_path(skeletal_mesh_path)

    # Get the LOD model
    lod_info = skeletal_mesh.get_editor_property('lod_info')
    if not lod_info:
        unreal.log_error(f"Could not get LOD info for Skeletal Mesh: {skeletal_mesh_path}")
        return

    # Get the sections from the LOD model
    sections = lod_info[0].get_editor_property('sections')

    for section_index, section in enumerate(sections):
        # Get the material slot name
        material_index = section.get_editor_property('material_index')
        material = skeletal_mesh.get_material(material_index)
        if not material:
            unreal.log_error(f"Could not get material for index: {material_index}")
            continue

        material_name = material.get_name()
        
        # Create a Static Mesh object to store the extracted part
        static_mesh_name = f"{skeletal_mesh.get_name()}_{material_name}"
        static_mesh = unreal.StaticMesh(None, static_mesh_name)
        
        # Create a new LOD for the static mesh
        lod = unreal.StaticMeshLODResources()
        static_mesh.add_editor_property('lods', [lod])
        
        # Add the mesh section from the skeletal mesh to the static mesh
        static_mesh.add_mesh_section(lod, section)
        
        # Set the material of the static mesh
        static_mesh.set_material(0, material)
        
        # Save the static mesh asset in the same directory as the skeletal mesh
        package_name = f"{skeletal_mesh_directory}/{static_mesh_name}"
        package_path = unreal.Paths.combine([skeletal_mesh_directory, static_mesh_name])
        if not unreal.EditorAssetLibrary.save_asset(package_path, static_mesh):
            unreal.log_error(f"Failed to save Static Mesh: {package_path}")
        else:
            unreal.log(f"Saved Static Mesh: {package_path}")

# Usage
skeletal_mesh_path = "/Game/NPC/Normal_Soldier/Common/Skins_V1/Skin_Weapon_GodrickSoldier_StraightSword.Skin_Weapon_GodrickSoldier_StraightSword"
extract_and_save_static_meshes_from_skeletal_mesh(skeletal_mesh_path)
