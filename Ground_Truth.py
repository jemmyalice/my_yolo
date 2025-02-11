import os
import cv2
import numpy as np

# 输入输出目录
image_dir = r'F:\yolo_change_try\ultralytics-main\data\VEDAI\VEDAI512_converted\visible\train\images'
label_dir = r'F:\yolo_change_try\ultralytics-main\data\VEDAI\VEDAI512_converted\visible\train\labels'
output_dir = r'F:\yolo_change_try\ultralytics-main\output'
# 类别名称
class_names = ['car', 'pickup', 'camping', 'truck', 'other', 'tractor', 'boat', 'van']

# 为每个类别选择与深红色比较匹配好看的颜色
colors = [
    (0, 0, 139),      # 深红色，与示例中保持一致
    (39, 39, 144),    # 类似深紫色，与深红搭配协调
    (71, 60, 139),    # 深靛蓝色
    (123, 104, 238),  # 蓝紫色
    (153, 50, 204),   # 深紫罗兰色
    (139, 0, 139),    # 深紫红色
    (148, 0, 211),    # 深靛紫色
    (186, 85, 211)    # 淡紫色
]

# 假设的置信度值
conf = 0.8

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

            # 绘制矩形框，调整颜色和粗细（将粗细改为 2）
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # 标签文本
            # label = f"{class_names[class_id]} {conf:.2f}" if class_id < len(class_names) else f"{class_id} {conf:.2f}"
            label = f"{class_names[class_id]}" if class_id < len(class_names) else f"{class_id}"

            # 调小字体大小和厚度
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # 调整标签位置，避免超出图片边界
            text_x = x_min
            text_y = y_min - 5
            if text_x + text_width > img_width:
                text_x = img_width - text_width
            if text_y - text_height < 0:
                text_y = text_height + 5

            # 绘制黑色背景矩形
            cv2.rectangle(image, (text_x, text_y - text_height), (text_x + text_width, text_y), (0, 0, 0), -1)

            # 在黑色背景上绘制白色文本
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 保存图片到输出文件夹
        output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_image_path, image)
        print(f"Processed {image_name} and saved to {output_image_path}")