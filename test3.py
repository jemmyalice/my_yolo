import warnings
from idlelib.configdialog import tracers
from PIL.ImageFont import truetype
from ultralytics import YOLO
import torch
warnings.filterwarnings('ignore')

# 加载模型（假设你的模型是在一个类中定义的）
# model = YOLO(r'F:\ultralytics-main\ultralytics\cfg\models\11\Fusion18_1212_yolo11s.yaml')
weights  = torch.load(r'C:\Users\86137\Downloads\best.pt', map_location=torch.device('cpu'))
# model.load_state_dict(weights)  # 加载权重
print(weights.keys())
# 从字典提取模型部分
model_state_dict = weights['model']  # 提取 'model' 部分
# 提取模型部分（实际上是模型对象）
model = weights['model']  # 此时 model 是一个 DetectionModel 的实例

# 打印模型结构
print(model)

# 查看各层参数的形状
for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")  # 输出每个参数的名称和形状

# 如果想查看具体参数值，可以这样：
for name, param in model.named_parameters():
    if param.size() == torch.Size([]):
        print(f"{name}: {param.data}")  # 输出每个参数的名称和对应的值