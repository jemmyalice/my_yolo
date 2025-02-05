import os
import cv2
import numpy as np

# 输入输出目录
image_dir = r'F:\yolo_change_try\ultralytics-main\data\VEDAI\VEDAI512_converted\visible\train\images'
label_dir = r'F:\yolo_change_try\ultralytics-main\data\VEDAI\VEDAI512_converted\visible\train\labels'
output_dir = r'F:\yolo_change_try\ultralytics-main\data\VEDAI\VEDAI512_converted\output'

# 类别名称
class_names = ['car', 'pickup', 'camping', 'truck', 'other', 'tractor', 'boat', 'van']

# 为每个类别选择一个颜色
# 颜色选择为中等色调，不太亮的颜色
colors = [
    (255, 0, 0),       # car: 红色 (鲜艳的红色，易于辨识)
    (0, 255, 0),       # pickup: 绿色 (亮绿色，能突出显示)
    (0, 0, 255),       # camping: 蓝色 (标准蓝色，清晰明亮)
    (255, 165, 0),     # truck: 橙色 (橙色在遥感图像中对比度高)
    (255, 255, 255),   # other: 白色 (高对比度，适合背景或其他类目标)
    (0, 255, 255),     # tractor: 亮青色 (明亮且鲜艳，能与多数背景区分开)
    (255, 0, 255),     # boat: 紫色 (亮紫色，显眼且与其他目标区分度高)
    (255, 255, 0)      # van: 黄色 (黄色具有高对比度，适合在遥感图像中显示)
]

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历图片文件夹
for image_name in os.listdir(image_dir):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        # 加载图片
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        # 对应的标签文件
        label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        # 读取标签文件并绘制框
        with open(label_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # YOLO 格式：类索引 x_center y_center width height
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 转换为图像坐标
            img_width, img_height = image.shape[1], image.shape[0]
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)

            # 获取类别的颜色
            color = colors[class_id]

            # 绘制矩形框
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # 在框上方写上类名
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 保存图片到输出文件夹
        output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_image_path, image)
        print(f"Processed {image_name} and saved to {output_image_path}")
