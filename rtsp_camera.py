#!/usr/bin/env python
import cv2, timeit

video = cv2.VideoCapture("rtsp://admin:tipa1234@192.168.90.243:554/cam/realmonitor?channel=1&subtype=0")
print('abriu camera!')

while (True):

    ############## Time ##############
    start = timeit.default_timer()

    # -- Capture frame-by-frame
    _, frame = video.read()
    # (rows,cols,channels) = frame.shape
    # print("shape: ", frame.shape) #720x1280x3

    # scale_percent = 50  # percent of original size
    # width_dnn = int(frame.shape[1] * scale_percent / 100)
    # height_dnn = int(frame.shape[0] * scale_percent / 100)
    # dim_dnn = (width_dnn, height_dnn)

    # imgResize = cv2.resize(frame,(320,240))
    # imgResize = cv2.resize(frame, dim_dnn)
    # print("shape: ", imgResize.shape)  # (540, 960, 3)
    # print(imgResize.shape)

    #gray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

    # -- Print img gray
    cv2.imshow('RTSP', frame)

    stop = timeit.default_timer()
    time_cascade = round((stop - start) * 1000, 1)
    print('Time:', time_cascade, 'ms')
    ############## Time ##############

    # print('#######################\n')

    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
