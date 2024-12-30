import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

def convert_voc_to_yolo(xml_file, img_width, img_height, label_map):
    """
    将 VOC 格式的 XML 标签文件转换为 YOLO 格式。
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in label_map:
            continue

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / float(img_width)
        height = (ymax - ymin) / float(img_height)

        class_id = label_map[class_name]
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return "\n".join(yolo_labels)

def process_dataset(voc_annotations_dir, infrared_dir, visible_dir, output_dir, label_map, train_ratio=0.8):
    """
    将 LLVIP 数据集分割为红外和可见光两个数据集，并转换标签为 YOLO 格式。
    """
    # 创建输出目录
    for mode in ['infrared', 'visible']:
        for subset in ['train', 'test']:
            os.makedirs(os.path.join(output_dir, mode, subset, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, mode, subset, 'labels'), exist_ok=True)

    # 获取所有 XML 文件及其对应图片文件
    all_xml_files = [f for f in os.listdir(voc_annotations_dir) if f.endswith('.xml')]
    all_img_files = [f.replace('.xml', '.jpg') for f in all_xml_files]  # 假设图片格式为 JPG

    # 分割训练集和测试集
    train_xml, test_xml = train_test_split(all_xml_files, train_size=train_ratio, random_state=42)

    # 处理每个分集
    for subset, xml_files in zip(['train', 'test'], [train_xml, test_xml]):
        for xml_file in xml_files:
            xml_path = os.path.join(voc_annotations_dir, xml_file)
            base_name = os.path.splitext(xml_file)[0]

            for mode, img_dir in [('infrared', infrared_dir), ('visible', visible_dir)]:
                img_file = os.path.join(img_dir, f"{base_name}.jpg")
                if not os.path.exists(img_file):
                    print(f"Image file {img_file} not found. Skipping.")
                    continue

                from PIL import Image
                img = Image.open(img_file)
                img_width, img_height = img.size

                # 转换为 YOLO 格式标签
                yolo_labels = convert_voc_to_yolo(xml_path, img_width, img_height, label_map)

                # 保存图片和标签
                output_image_path = os.path.join(output_dir, mode, subset, 'images', f"{base_name}.jpg")
                output_label_path = os.path.join(output_dir, mode, subset, 'labels', f"{base_name}.txt")
                shutil.copy(img_file, output_image_path)
                with open(output_label_path, 'w') as f:
                    f.write(yolo_labels)

# 标签映射字典（根据实际数据集修改）
label_map = {
    'class_name_1': 0,
    'class_name_2': 1,
    # 更多类别...
}

# 文件路径（根据实际路径调整）
voc_annotations_dir = 'LLVIP/VOC/Annotations'
infrared_dir = 'LLVIP/inf/train'
visible_dir = 'LLVIP/images/train'
output_dir = 'LLVIP_converted'

# 调用函数
process_dataset(voc_annotations_dir, infrared_dir, visible_dir, output_dir, label_map)
