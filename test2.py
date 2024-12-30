from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 加载两张图像
image_pathir = r'F:\ultralytics-main\data\VEDAI\VEDAI512_converted\infrared\test\images\00001237.png'  # 替换为第一张图像的路径
image_pathvi = r'F:\ultralytics-main\data\VEDAI\VEDAI512_converted\visible\test\images\00001237.png'  # 替换为第二张图像的路径
# image_pathir = r'F:\ultralytics-main\data\llvip\LLVIP_converted\infrared\test\images\260433.jpg'  # 替换为第一张图像的路径
# image_pathvi = r'F:\ultralytics-main\data\llvip\LLVIP_converted\visible\test\images\260433.jpg'  # 替换为第二张图像的路径



# 打开图像并转换为 RGB
image1 = Image.open(image_pathir).convert('RGB')
image2 = Image.open(image_pathvi).convert('RGB')

# 将图像转换为 NumPy 数组
array1 = np.array(image1)
array2 = np.array(image2)

# 检查图像的形状是否相同
if array1.shape != array2.shape:
    raise ValueError("两张图像的大小和通道数必须相同！")

# 计算相减结果
diff_array = array1*0.5 - array2*0.5
diff_array1 = array2*0.5 - array1*0.5

# 计算加上相减结果后的新图像
new_image1 = array1 + diff_array
new_image2 = array2 + diff_array1

# 限制值在 0 到 255 之间，以防有任何溢出
new_image1 = np.clip(new_image1, 0, 255).astype(np.uint8)
new_image2 = np.clip(new_image2, 0, 255).astype(np.uint8)

# 设定显示图像的大小
plt.figure(figsize=(12, 10))

# 显示第一张图像
plt.subplot(2, 3, 1)
plt.title("Image 1")
plt.imshow(image1)
plt.axis('off')

# 显示第二张图像
plt.subplot(2, 3, 2)
plt.title("Image 2")
plt.imshow(image2)
plt.axis('off')

# 显示相减的结果
plt.subplot(2, 3, 3)
plt.title("Difference (Image 1 - Image 2)")
plt.imshow(np.clip(diff_array, 0, 255).astype(np.uint8))  # 限制在 0-255 之间
plt.axis('off')

# 显示第一张图像加上相减结果
plt.subplot(2, 3, 4)
plt.title("Image 1 + Difference")
plt.imshow(new_image1)
plt.axis('off')

# 显示第二张图像加上相减结果
plt.subplot(2, 3, 5)
plt.title("Image 2 + Difference")
plt.imshow(new_image2)
plt.axis('off')

# 调整布局
plt.tight_layout()
plt.show()