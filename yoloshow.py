import os
import cv2  # OpenCV 用于图像处理
import numpy as np


def draw_yolo_boxes(img_path, label_path, output_path, class_names=None):
    """
    根据 YOLO 格式标签绘制边界框并保存图片。
    :param img_path: 图像路径
    :param label_path: YOLO 标签路径
    :param output_path: 输出图片路径
    :param class_names: 类别名称列表（可选）
    """
    # 加载图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image {img_path}")
        return

    height, width, _ = img.shape

    # 读取 YOLO 标签文件
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return

    # 绘制每个边界框
    for line in lines:
        elements = line.strip().split()
        if len(elements)!=5:
            print(f"Error in label format: {line}")
            continue

        class_id, x_center, y_center, box_width, box_height = map(float, elements)
        x_center, y_center, box_width, box_height = (
            x_center * width,
            y_center * height,
            box_width * width,
            box_height * height,
        )

        # 计算边界框左上角和右下角坐标
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # 设置边界框颜色
        color = (0, 255, 0)  # 绿色
        label_text = str(int(class_id))
        if class_names and int(class_id) < len(class_names):
            label_text = class_names[int(class_id)]

        # 绘制矩形框和标签
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")


# 示例：验证 YOLO 格式标签
image_dir = r"F:\ultralytics-main\LLVIP\yolo_labels\train"  # 图片路径
label_dir = r"F:\ultralytics-main\LLVIP\yolo_labels\train"  # YOLO 标签路径
output_dir = r"F:\ultralytics-main\LLVIP\yolo_labels\train"  # 输出图片路径
class_names = ['person']  # 自定义类别名称

for img_file in os.listdir(image_dir):
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

        # 修改输出文件名，加上 "_with_boxes"
        output_file_name = os.path.splitext(img_file)[0] + '_with_boxes' + os.path.splitext(img_file)[1]
        output_path = os.path.join(output_dir, output_file_name)

        # 检查标签文件是否存在
        if os.path.exists(label_path):
            draw_yolo_boxes(img_path, label_path, output_path, class_names)
        else:
            print(f"Label file not found for {img_file}. Skipping.")
