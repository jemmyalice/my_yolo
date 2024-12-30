import os
import xml.etree.ElementTree as ET
import shutil

def convert_voc_to_yolo(xml_file, img_width, img_height, label_map):
    """
    将 VOC 格式的 XML 标签文件转换为 YOLO 格式。
    :param xml_file: VOC 格式的 XML 文件路径
    :param img_width: 图像的宽度
    :param img_height: 图像的高度
    :param label_map: 标签映射字典
    :return: YOLO 格式标签字符串
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return None

    yolo_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in label_map:
            print(f"Class '{class_name}' not found in label map, skipping.")
            continue

        # 获取边界框（xmin, ymin, xmax, ymax）
        bndbox = obj.find('bndbox')
        try:
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
        except Exception as e:
            print(f"Error reading bounding box in file {xml_file}: {e}")
            continue

        # 计算 YOLO 格式的标签，归一化到 [0, 1] 区间
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / float(img_width)
        height = (ymax - ymin) / float(img_height)

        # 获取类标签（YOLO 格式中类别从0开始）
        class_id = label_map[class_name]

        # 将结果存储为 YOLO 格式
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return "\n".join(yolo_labels)

def process_images_and_labels(voc_annotations_dir, image_dir, output_dir, label_map):
    """
    遍历图像文件夹和标签文件夹，转换为 YOLO 格式标签文件。
    :param voc_annotations_dir: VOC 标签文件夹路径
    :param image_dir: 图像文件夹路径
    :param output_dir: 输出目录路径
    :param label_map: 标签映射字典
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            img_path = os.path.join(image_dir, img_file)
            xml_file = os.path.join(voc_annotations_dir, os.path.splitext(img_file)[0] + '.xml')

            # 检查图片和标签是否存在
            if not os.path.exists(img_path):
                print(f"Image file {img_path} not found. Skipping.")
                continue
            if not os.path.exists(xml_file):
                print(f"XML file {xml_file} not found for image {img_file}. Skipping.")
                continue

            try:
                # 获取图像的宽度和高度
                from PIL import Image
                img = Image.open(img_path)
                img_width, img_height = img.size

                # 转换为 YOLO 格式标签
                yolo_labels = convert_voc_to_yolo(xml_file, img_width, img_height, label_map)

                if yolo_labels is None or yolo_labels.strip() == "":
                    print(f"No valid labels generated for {img_file}. Skipping.")
                    continue

                # 保存 YOLO 格式的标签文件
                yolo_label_file = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.txt')
                with open(yolo_label_file, 'w') as f:
                    f.write(yolo_labels)

                # 复制图像到输出文件夹（如果需要）
                shutil.copy(img_path, os.path.join(output_dir, img_file))
                print(f"Processed {img_file} successfully.")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
# 标签映射字典（根据你的数据集调整）
label_map = {
    'person': 0,
    # 更多类别...
}

# VOC 标签目录、训练/测试图像目录和输出目录
voc_annotations_dir = r'F:\python_anaconda\envs\myenv\My_data\labels'
image_dir = r'F:\python_anaconda\envs\myenv\My_data\images'  # 或者 'LLVIP/test' 取决于你要处理的是训练集还是测试集
output_dir = 'LLVIP/yolo_labels/train'  # 输出 YOLO 格式标签的目录

process_images_and_labels(voc_annotations_dir, image_dir, output_dir, label_map)
