import numpy as np
import rasterio
from PIL import Image


def shuchu(tif_file):
    """
    处理TIFF遥感图像，生成真彩色归一化图像

    参数:
        tif_file (str): 输入的TIFF文件路径

    返回:
        numpy.ndarray: 形状为(height, width, 3)的uint8数组，表示归一化的RGB图像
    """
    # 打开TIFF文件
    with rasterio.open(tif_file) as src:
        # 读取所有波段（假设波段顺序为B02, B03, B04, B08, B12）
        bands = src.read()

    # 提取波段并转换为浮点数类型
    blue = bands[0].astype(float)  # B02 - 蓝
    green = bands[1].astype(float)  # B03 - 绿
    red = bands[2].astype(float)  # B04 - 红

    # 创建RGB组合
    rgb_orign = np.dstack((red, green, blue))

    # 计算最小值和最大值用于归一化
    array_min, array_max = rgb_orign.min(), rgb_orign.max()

    # 归一化到0-255范围
    rgb_normalized = ((rgb_orign - array_min) / (array_max - array_min)) * 255
    rgb_normalized = rgb_normalized.astype(np.uint8)

    return rgb_normalized


# 使用示例
if __name__ == "__main__":
    # 输入TIFF文件路径
    input_tif = r"D:\shixi\2019_1101_nofire_B2348_B12_10m_roi.tif"

    # 输出图像路径
    output_image = r"D:\shixi\output_rgb.jpg"

    try:
        # 处理TIFF文件
        result = shuchu(input_tif)

        # 保存结果
        Image.fromarray(result).save(output_image)
        print(f"成功生成真彩色图像并保存至: {output_image}")

        # 可选：显示图像
        # import matplotlib.pyplot as plt
        # plt.imshow(result)
        # plt.axis('off')
        # plt.show()

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")