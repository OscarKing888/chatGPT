import unreal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_all_textures():
    """
    从Unreal Engine项目中加载所有的贴图。
    返回一个包含贴图数据的列表。
    """
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    texture_assets = asset_registry.get_assets_by_class('Texture2D')
    
    textures = []
    for asset_data in texture_assets:
        texture = unreal.EditorAssetLibrary.load_asset(asset_data.get_full_name())
        textures.append(texture)
    
    return textures

def convert_to_grayscale(texture):
    """
    将贴图转换为灰度图。
    """
    texture_data = unreal.Texture2DHelpers.get_texture_data(texture)
    texture_array = np.array(texture_data, dtype=np.float32)
    if texture_array.ndim == 3 and texture_array.shape[2] == 4:
        # 如果是RGBA格式，转为灰度
        grayscale_texture = 0.2989 * texture_array[:,:,0] + 0.5870 * texture_array[:,:,1] + 0.1140 * texture_array[:,:,2]
    else:
        grayscale_texture = texture_array
    return grayscale_texture

def extract_features(grayscale_texture):
    """
    提取灰度图的特征，可以使用直方图。
    """
    histogram, _ = np.histogram(grayscale_texture, bins=256, range=(0, 256))
    histogram = histogram.astype(np.float32)
    histogram /= np.sum(histogram)  # 归一化
    return histogram

def calculate_similarity(features1, features2):
    """
    计算两个特征向量之间的余弦相似度。
    """
    similarity = cosine_similarity([features1], [features2])
    return similarity[0][0]

def find_similar_textures(textures, threshold=0.8):
    """
    找到相似度接近80%的贴图对。
    """
    similar_textures = []
    features_list = [extract_features(convert_to_grayscale(texture)) for texture in textures]
    
    for i in range(len(features_list)):
        for j in range(i + 1, len(features_list)):
            similarity = calculate_similarity(features_list[i], features_list[j])
            if 0.75 <= similarity <= 0.85:  # 设置一个范围来近似80%
                similar_textures.append((textures[i].get_name(), textures[j].get_name(), similarity))
    
    return similar_textures

def main():
    textures = load_all_textures()
    similar_textures = find_similar_textures(textures)
    
    for tex1, tex2, similarity in similar_textures:
        print(f"Texture 1: {tex1} and Texture 2: {tex2} have similarity: {similarity}")

if __name__ == "__main__":
    main()
