import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("./yolov8model/test/Traffic_best_20Nov.pt")

print(model.names)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    print(frame.shape)

    dim = (640, 640)

    # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
 
    print('Resized Dimensions : ',frame.shape)
    results = model(frame,  device="mps")
    result = results[0]
    bboxes = result.boxes.xyxy

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(model.names[int(cls)]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    print(bboxes)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()