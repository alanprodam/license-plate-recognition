#!/usr/bin/env python
import cv2

video = cv2.VideoCapture("rtsp://admin:tipa1234@saojoaquim.letmein.com.br:554/cam/realmonitor?channel=1&subtype=0")
print('abriu camera!')

while (True):
    # -- Capture frame-by-frame
    _, frame = video.read()
    # (rows,cols,channels) = frame.shape
    # print(frame.shape) #720x1280x3

    # imgResize = cv2.resize(frame,(320,240))
    imgResize = cv2.resize(frame, (640, 480))
    # print(imgResize.shape)

    gray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

    # -- Print img gray
    cv2.imshow('RTSP', gray)

    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
