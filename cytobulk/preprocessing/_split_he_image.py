import openslide
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from .. import utils

def process_svs_image(svs_path, output_dir, project, crop_size=224, magnification=20):
    utils.check_paths(f'{output_dir}/output')

    # 读取SVS文件
    slide = openslide.OpenSlide(svs_path)
    
    # 获取图像的宽和高
    width, height = slide.dimensions
    print(f"Original image size: {width}x{height}")
    
    # 计算图像中心坐标
    center_x, center_y = width // 2, height // 2
    print(f"Image center: ({center_x}, {center_y})")
    
    # 截取一个部分（从中心坐标开始，选一个2000x2000的区域）
    crop_width, crop_height = 2000, 2000
    start_x = max(center_x - crop_width // 2, 0)
    start_y = max(center_y - crop_height // 2, 0)
    print(f"Crop region: Start=({start_x}, {start_y}), Size=({crop_width}, {crop_height})")
    
    # 读取截取的区域
    region = slide.read_region((start_x, start_y), 0, (crop_width, crop_height))
    region = region.convert("RGB")  # 转换为RGB模式
    
    # 显示截取的图像
    plt.imshow(region)
    plt.title("Cropped Region")
    plt.axis("off")
    plt.show()
    
    # 将截取部分放大20倍
    enlarged_width = crop_width * magnification
    enlarged_height = crop_height * magnification
    region_enlarged = region.resize((enlarged_width, enlarged_height), Image.LANCZOS)
    print(f"Enlarged image size: {region_enlarged.size}")
    
    # 从放大后的图像中裁剪224x224的图像
    for x in range(0, enlarged_width, crop_size):
        for y in range(0, enlarged_height, crop_size):
            # 确保裁剪区域不超出图像范围
            if x + crop_size <= enlarged_width and y + crop_size <= enlarged_height:
                cropped = region_enlarged.crop((x, y, x + crop_size, y + crop_size))
                
                # 保存裁剪后的图像，命名为最小的XY坐标
                filename = f"{x}_{y}.jpg"
                cropped.save(os.path.join(output_dir, filename))
                print(f"Saved: {filename}")

if __name__ == "__main__":
    # 输入SVS文件路径
    svs_file_path = "path/to/your/file.svs"  # 替换为你的SVS文件路径
    output_directory = "output_images"  # 输出文件夹
    
    # 运行处理函数
    process_svs_image(svs_file_path, output_directory)