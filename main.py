from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load YOLOv8 pretrained hand detection model
model = YOLO("yolov8n.pt")  # nano version for speed

# Load your keypoint regression CNN
keypoint_model = torch.load("hand_kp_model.pt")
keypoint_model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    # Step 1: Detect hands
    results = model(frame)
    for box in results[0].boxes.xyxy:  # x1,y1,x2,y2
        x1, y1, x2, y2 = map(int, box)
        hand_crop = frame[y1:y2, x1:x2]

        # Step 2: Predict keypoints
        kp_input = preprocess(hand_crop)  # resize + normalize
        kp_output = keypoint_model(kp_input)
        keypoints = kp_output.detach().numpy().flatten()  # 21 x 2

        # Step 3: Compare to reference keypoints
        score = compute_score(keypoints, reference_kp)
        print("Similarity score:", score)

    cv2.imshow("Hand Sign", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
