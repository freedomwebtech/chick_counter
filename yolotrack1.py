import cv2
import numpy as np
from ultralytics import YOLO
import cvzone


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# Load COCO class names
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('vid.mp4')
count=0
# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
        # Loop through the detected boxes and their associated class IDs and track IDs
        chickscounter=[]
        chickencounter=[]
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            
            x1, y1, x2, y2 = box
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            if 'chick' in c:
               cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
               cvzone.putTextRect(frame, f'{track_id}', (cx,cy), 1, 1)
               chickscounter.append(track_id)
            if 'chicken' in c:
               cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               cvzone.putTextRect(frame, f'{track_id}', (cx,cy), 1, 1)
               chickencounter.append(track_id)          

               
        counterch=len(chickscounter)
        chickench=len(chickencounter)
        cvzone.putTextRect(frame, f'ChicksCounter:-{counterch}', (50,60), 1, 1)
        cvzone.putTextRect(frame, f'ChickenCounter:-{chickench}', (50,160), 1, 1)

        cv2.imshow("RGB", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
