import cv2
import time
cpt = 0
maxFrames = 100 # if you want 5 frames only.

cap=cv2.VideoCapture('chicks-and-mother-are-eating.mp4')
count=0
while cpt < maxFrames:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
    frame=cv2.resize(frame,(1020,600))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"C:\Users\freed\Downloads\yolov8peoplecounter-main\yolov8peoplecounter-main\images\img_%d.jpg" %cpt, frame)
    cpt += 1
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
