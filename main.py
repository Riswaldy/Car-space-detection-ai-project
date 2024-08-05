import cv2
import torch
import numpy as np
import pickle
import cvzone

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load parking positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# Dimensions of parking spaces
width, height = 107, 48

def checkParkingSpace(img, posList):
    spaceCounter = 0

    for pos in posList:
        x, y = pos
        imgCrop = img[y:y+height, x:x+width]

        # Convert imgCrop to RGB for YOLOv5
        imgCrop_rgb = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
        result = model(imgCrop_rgb)
        detected_objects = result.pandas().xyxy[0]

        # Check if any car is detected in the cropped image
        if len(detected_objects[(detected_objects['name'].isin(['car', 'truck', 'bus']))]) == 0:
            color = (0, 255, 0)  # Green color for free space
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red color for occupied space
            thickness = 2

        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, thickness)

    cvzone.putTextRect(img, f'Free: {spaceCounter} / {len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

    return img

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Run YOLOv5 model
    results = model(img)

    # Check parking spaces
    img = checkParkingSpace(img, posList)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    # Check if the 'q' key is pressed to quit
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
