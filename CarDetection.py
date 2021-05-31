import os
import time
import requests
import json
import cv2

from ctypes import c_bool

import config
from configs import config_application


def network_infer(frame, dim, net, w, h, conf=0.6):
    blob = cv2.dnn.blobFromImage(frame, size=dim, ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    cars = []
    coord = -1
    for detection in out.reshape(-1, 7):
        classId = int(detection[1])
        confidence = float(detection[2])
        if confidence > conf and classId == 1:
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)

            car_coord = (xmin, ymin, xmax - xmin, ymax - ymin)
            cars.append(car_coord)
            coord = xmin, xmax, ymin, ymax
            # point_center = calculate_centr(coord)
        else:
            break
    return coord


def plateCar(frameCropped):
    url = 'https://lpr.letmein.com.br/upload'

    _, imencoded = cv2.imencode(".jpg", frameCropped)
    file = {'file': ('lpr.jpg', imencoded)}
    r = requests.post(url, files=file)
    data = json.loads(r.text)

    # print(data)
    # print(data[0]['plate'])
    # print(data[0]['possiblePlate'])
    # print(data[0]['probabilities'])
    # print(data[0]['width'])
    # print(data[0]['x'])
    # print(data[0]['y'])
    result = data[0]['possiblePlate']

    return result


class CarDetection:
    def __init__(self):
        self.min_car_height = config_application.min_size
        self.max_car_height = config_application.max_size

        # Neural network for face identification
        self.net = cv2.dnn.readNet(
            'data/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.bin',
            'data/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml')
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)

        self.ignore_left_margin = int(1080 * config_application.ignore_left_margin)
        self.ignore_right_margin = int(1080 * config_application.ignore_left_margin)

    def detect_car(self, frame_to_process):
        print("CarDetection running!")
        # keep looping indefinitely until the thread is stopped

        # scale_percent = 25  # percent of original size
        width_orig = 1080
        height_orig = 1920
        # width = int(width_orig * scale_percent / 100)
        # height = int(height_orig * scale_percent / 100)
        dim = (512, 512)

        cars = network_infer(frame_to_process, (512, 512), self.net, width_orig, height_orig)

        tracks_position = []
        for (x, y, w, h) in cars:
            tracks_position = x, y, w, h
            print('tracks_position: ', tracks_position)

        print("CarDetection done.")

        return tracks_position
