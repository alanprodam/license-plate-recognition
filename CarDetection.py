import os
import time
import timeit
import datetime
from multiprocessing import Process, Queue

import cv2

import config
from configs import config_application


def network_infer(frame, dim, net, w, h, conf=0.6):
    blob = cv2.dnn.blobFromImage(frame, size=dim, ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    cars = []
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
            # coord = xmin, xmax, ymin, ymax
            # point_center = calculate_centr(coord)
        else:
            break
    return cars

class CarDetection:
    def __init__(self):
        self.max_queue_size = config_application.max_queue_size
        self.frame_queue = Queue(maxsize=config_application.max_queue_size)
        self.process = Process(target=self.detect_car, args=())
        self.process.daemon = True
        self.last_frame = []

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
        # 0 -> Esquerda, 1 -> Direita
        # Detectar face no frame especificado
        self.process_frame = config.process_frame

        self.track_cars = []
        self.tracked_cars_queue = Queue(maxsize=config_application.max_queue_size)
        # self.tracked_cars_queue_recognition = Queue(maxsize=config_application.max_queue_size)
        self.max_track_size = config_application.max_track_size
        self.min_frames_before_recognition = config_application.min_frames_before_recognition
        self.track_life = config_application.track_life_in_seconds

        self.process.start()

    def detect_car(self):
        print("CarDetection running! pid:", os.getpid())
        # keep looping indefinitely until the thread is stopped

        # scale_percent = 25  # percent of original size
        width_orig = 1920
        height_orig = 1080
        # width = int(width_orig * scale_percent / 100)
        # height = int(height_orig * scale_percent / 100)
        dim = (512, 512)

        skip_frame = False
        skip_N_frames = False
        N_counter = 0
        N = 3
        frames_counter = 0
        while True:
            frames_counter += 1
            # read the queue
            frame_left, frame_right, stop_process = self.frame_queue.get()

            # if the indicator variable is set, stop the thread
            if stop_process:
                print("CarDetection stopping...")
                time.sleep(0.1)  # wait 100 ms (wait queues to settle)
                break

            if skip_N_frames:
                if N_counter < N - 1:
                    N_counter += 1
                    continue
                else:
                    N_counter = 0
                    skip_N_frames = False
                    skip_frame = True
            else:
                skip_frame = not skip_frame
                if skip_frame:
                    # print(frames_counter, '- skipped')
                    continue

            # additional frame validation
            if frame_left is None:
                time.sleep(0.025)
                print('detect_face - frame is None')
                continue

            if self.process_frame == 0:
                frame_to_process = frame_left
            else:
                frame_to_process = frame_left

            cars = network_infer(frame_to_process, dim, self.net, width_orig, height_orig)

            tracks_found = []
            # for (x, y, w, h) in cars:
            #     # ignore small and big faces (probably noise)
            #     if h < self.min_face_height or h > self.max_face_height:
            #         # print('Face ignored due to size. Face height size:', h, 'Min and Max allowed:', self.min_face_height, self.max_face_height)
            #         continue
            #     cx = int(x + w / 2)
            #     if cx < self.ignore_left_margin or cx > self.ignore_right_margin:
            #         # print('Face próximo à margem. Ignorada.')
            #         continue
            #     current_face = Faces(x, y, w, h, timeit.default_timer())
            #     tracks_found.append(self.fill_list(current_face, frame_left, frame_right))
            #
            # self.update_tracks_main(tracks_found)  # to display in the main process
            #
            # tracks_to_send = []
            # for track in tracks_found:
            #     if len(track.list) >= self.min_frames_before_recognition:
            #         tracks_to_send.append(track)
            #
            # if len(tracks_to_send) != 0:
            #     tracks_to_send_sorted = sorted(tracks_to_send, key=lambda x: x.median_size_bbox, reverse=True)
            #     self.update_tracks(tracks_to_send_sorted)
            #     skip_N_frames = True

        print("CarDetection done.")
        return