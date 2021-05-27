import cv2

cap = cv2.VideoCapture('/home/alan/dataset_letmein/radar_1920x1080/dataset_radar_5.avi')
success_frame, frame = cap.read()

if success_frame:
    cv2.imwrite("../testImgs/BSZ6278.jpg", frame)
    cv2.imshow('frame', frame)

    cv2.waitKey(0)