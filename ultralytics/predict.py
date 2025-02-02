import warnings
from idlelib.configdialog import tracers
from PIL.ImageFont import truetype
from ultralytics import YOLO
import torch.nn as nn
import cv2
import torch
warnings.filterwarnings('ignore')
# 25 31
if __name__=='__main__':
    # new_model = YOLO(r'C:\Users\86137\Downloads\best (4).pt')
    # # 指定包含图像的文件夹路径
    # source = r'F:\yolo_change_try\ultralytics-main\data\VEDAI\VEDAI512_converted\visible\test\images'
    #
    # # 进行批量预测
    # # results = new_model(source)
    # result = new_model(r'F:\yolo_change_try\ultralytics-main\data\VEDAI\VEDAI512_converted\visible\test\images\00001237.png')

    def predict(chosen_model, img, classes=[], conf=0.5):
        if classes:
            results = chosen_model.predict(img, classes=classes, conf=conf)
        else:
            results = chosen_model.predict(img, conf=conf)

        return results


    def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = predict(chosen_model, img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                    (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                    (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        return img, results


    model = YOLO(r'C:\Users\86137\Downloads\best (4).pt')

    # read the image
    image = cv2.imread(r"F:\yolo_change_try\ultralytics-main\data\VEDAI\VEDAI512_converted\visible\test\images\00001237.png")
    result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)

    cv2.imshow("Image", result_img)
    cv2.imwrite("YourSavePath.png", result_img)
    cv2.waitKey(0)