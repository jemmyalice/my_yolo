import warnings
from idlelib.configdialog import tracers
from PIL.ImageFont import truetype
from ultralytics import YOLO
import torch.nn as nn
import torch
warnings.filterwarnings('ignore')
# 25 31
if __name__=='__main__':
    # model = YOLO(r'F:\downloads\chorme\epoch80.pt')
    # # model = YOLO(r'F:\ultralytics-main\ultralytics\cfg\models\11\Fusion1_1212_yolo11s.yaml')
    # # model.train(
    # #     data=r'F:\ultralytics-main\data\llvip\data_infusion.yaml',  # Specify your dataset configuration
    # #     lr0=0.001,  # Learning rate
    # #     epochs=180,
    # #     save_period=20,
    # #     cos_lr=False,
    # #     resume=True,
    # #     multi_scale=False,
    # #     device='cpu'
    # # )
#22 23 24 25 :49 47 49 47   13版本、10版本、12版本、14版本+bn+weight
#22_1 22_2:29 45 13版本去除bn、weight

    # 22_2 是去18基础去norm 29
    # 22 是22——2基础上加tff 29
    model = YOLO(r'F:\yolo_change_try\ultralytics-main\ultralytics\cfg\models\11\2.5Fusion13_1212_yolo11s.yaml')
    # model = YOLO(r'F:\ultralytics-main\ultralytics\cfg\models\11\SKbase11_fusion_yolo11s.yaml')
    # model = YOLO(r'F:\yolo_change_try\ultralytics-main\ultralytics\cfg\models\11\Ayolo11s.yaml')

    state_dict = model.model.state_dict()
    i = 0
    for (k1, v1) in state_dict.items():
        # if "weight1" in k1 or "weight2" in k1:
        #     continue
        # if i in [13, 14, 31, 32]:
        #     print("顶顶顶顶顶顶顶顶顶顶顶顶顶顶")
        #     if "weight" in k1 and v1.dim() >= 2:  # 对权重进行初始化
        #         print("yes\n")
        #     elif "bias" in k1:  # 对偏置进行零初始化
        #         print("yes\n")
        #     elif "running_var" in k1:  # 对 gamma 进行初始化
        #         print("yes\n")
        #
        #     elif "running_mean" in k1:  # 对 beta 进行初始化
        #         print("yes\n")
        #
        #     if 'catconv' in k1 and 'weight' in k1 and v1.dim() >= 2:
        #         print("yes\n")

        i = i + 1
        if i==50:
            break
        pass
        print(f'{i} Name1:{k1} Size: {v1.numel()}')


    model.train(data=r'F:\ultralytics-main\data\VEDAI\data_infusion.yaml',
        lr0=0.0001,  # Learning rate
        plots = True,
        imgsz=640,  # Image size
        epochs=2,
        device='cpu',
        patience = 20, # 20轮性能没改善停止data=r'F:\ultralytics-main\data\llvip\data_infusion.yaml',
        batch=2,
        single_cls=False,  # 是否是单类别检测
        workers=0, # 设置0让他自己判断
        # cos_lr = True, # 学习率以cos曲线变化，论文中大多没有使用它
        # resume = True, # 从最后一个检查点恢复训练，具体不清楚
        # optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
        optimizer='Adamw',  # using SGD 优化器 默认为auto建议大家使用固定的.
        fraction =0.2, # 在跑通前用于自己测试
        exist_ok = True, # 在跑通前用于自己测试
        multi_scale = False, # 用于增加泛化性，但是会增加训练时常，要注意
        amp=True,  # 如果出现训练损失为Nan可以关闭amp
        save_period=20,  # 20轮保存一个pt，方便下次继续训练
        project='runs/train',
        name='exp666',
    )