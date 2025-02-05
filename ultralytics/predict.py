import warnings
from ultralytics import YOLO
import os
warnings.filterwarnings('ignore')

new_model = YOLO(r'C:\Users\86137\Downloads\15VER.pt')

results = new_model.val(
    data=r'F:\yolo_change_try\ultralytics-main\data\VEDAI\data_infusion.yaml',  # 配置文件路径，包含数据集路径和类别信息
    imgsz=640,  # 输入图像的大小，和训练时的一致
    batch=1,  # 每次评估时的批次大小
    device='cpu',  # 使用的设备，如果有 GPU 可设置为 'cuda'
    save_txt = True,
    save_hybrid = True
)