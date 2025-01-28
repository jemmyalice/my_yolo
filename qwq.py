from skimage import io, transform
import numpy as np
import os

def transform_image(input_image_path, output_image_path):
    image = io.imread(input_image_path)
    height, width = image.shape[:2]

    # 源图像的四个角点（左上，右上，右下，左下）
    src_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    # 调整侧视效果的参数
    offset_x = int(width * 0.2)  # 可根据需要调整
    target_points = np.array([[0, 0], [width, 0], [width + offset_x, height], [0, height]], dtype=np.float32)

    # 计算仿射变换矩阵
    matrix = transform.AffineTransform(from_points=src_points, to_points=target_points).params[:2, :]

    # 应用仿射变换
    transformed_image = transform.warp(image, inverse_map=np.linalg.inv(matrix), output_shape=(height, width + offset_x))

    # 转换为合适的数据类型并保存
    transformed_image = (transformed_image * 255).astype(np.uint8)
    io.imsave(output_image_path, transformed_image)

# 定义图片文件夹路径
image_folder = r'C:\Users\86137\Desktop\Data\image'
# 定义输出文件夹路径
output_folder = os.path.join(image_folder, 'output')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹下的所有图片文件
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_image_path = os.path.join(image_folder, filename)
        output_filename = f'output_{filename}'
        output_image_path = os.path.join(output_folder, output_filename)
        transform_image(input_image_path, output_image_path)