import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os

# XML和图像文件的路径
xml_path = r'F:\python_anaconda\envs\myenv\My_data\190001.xml'
image_path = r'F:\python_anaconda\envs\myenv\My_data\190001.jpg'
output_image_path = r'F:\python_anaconda\envs\myenv\My_data\190001_with_boxes.jpg'

# 打印路径以进行调试
print(f"XML file path: {xml_path}")
print(f"Image file path: {image_path}")

# 检查文件是否存在
if not os.path.exists(xml_path):
    print(f"Error: XML file does not exist at {xml_path}")
    exit(1)
if not os.path.exists(image_path):
    print(f"Error: Image file does not exist at {image_path}")
    exit(1)

# 解析XML文件
try:
    tree = ET.parse(xml_path)
    root = tree.getroot()
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# 打开图像
try:
    image = Image.open(image_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# 绘制识别框
draw = ImageDraw.Draw(image)

# 假设识别框在XML中是以<object>标签包围的, <bndbox> 标签内包含坐标
for obj in root.findall('.//object'):
    bndbox = obj.find('bndbox')
    if bndbox is not None:
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 绘制矩形框
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)

# 保存带框的图像
image.save(output_image_path)
image.show()

print(f"Image with bounding boxes saved to {output_image_path}")
# import os
#
# # 设置要查看的文件夹路径
# folder_path = r'F:\python_anaconda\envs\myenv\include'
#
# # 检查路径是否存在
# if not os.path.exists(folder_path):
#     print(f"Error: The folder does not exist at the specified path: {folder_path}")
# else:
#     # 如果路径存在，列出文件
#     files = os.listdir(folder_path)
#     print(f"Files in folder {folder_path}: {files}")
