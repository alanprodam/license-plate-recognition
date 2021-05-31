import cv2
import timeit, time
import InputStreamReader as Sr
import os
import datetime
import config
from configs import config_application
import CarDetection as Cd
from multiprocessing.managers import BaseManager


import imutils
import requests
import json

color_black = (0, 0, 0)
color_blue = (255, 13, 13)
color_red = (13, 13, 210)
color_green = (13, 210, 13)
color_white = (255, 255, 255)

# color_red = (0, 0, 255)
# color_blue = (255, 0, 0)
rectThinkness = 2

spcial_char_map = {ord('á'): 'a', ord('ã'): 'a', ord('â'): 'a',
                   ord('é'): 'e', ord('ê'): 'e',
                   ord('í'): 'i',
                   ord('ó'): 'o', ord('õ'): 'o',
                   ord('ú'): 'u',
                   ord('ç'): 'c'
                   }

_print = print


def log_print(*args, **options):
    _print(datetime.datetime.now().strftime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), end='   ')
    _print(*args, **options)


# Change the print function globally
import builtins

builtins.print = log_print


def show_camera_fail(background_fail):
    cv2.imshow('Zion', background_fail)
    cv2.waitKey(500)

def calculate_centr_cut(coord):
    return int(abs(coord[0] - coord[1]) / 2 + coord[0]), int(abs(coord[2] - coord[3]) / 2 + coord[2])


def network_infer(frame, net, conf=0.9):
    blob = cv2.dnn.blobFromImage(frame, size=(512, 512), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()

    w = frame.shape[1]
    h = frame.shape[0]

    cars = []
    car_coord = -1
    point_center = -1
    for detection in out.reshape(-1, 7):
        classId = int(detection[1])
        confidence = float(detection[2])
        if confidence > conf and classId == 1:
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)
            car_coord = xmin, ymin, xmax, ymax
            cars.append(car_coord)
            center_coord = xmin, xmax, ymin, ymax
            point_center = calculate_centr_cut(center_coord)
        else:
            break
    return car_coord, point_center


def plateCar(frame):
    url = 'https://lpr.letmein.com.br/upload'


    _, imencoded = cv2.imencode(".jpg", frame)
    file = {'file': ('lpr.jpg', imencoded)}
    r = requests.post(url, files=file)
    data = json.loads(r.text)


    xmin = data[0]['x'] - data[0]['width']
    ymin = data[0]['y'] - data[0]['width']
    xmax = data[0]['x'] + data[0]['width']
    ymax = data[0]['y'] + data[0]['width']
    # result = data[0]['plate']
    result = data[0]['possiblePlate']
    cv2.waitKey(0)

    return result


def main():
    # Neural network for face identification
    net = cv2.dnn.readNet(
        'data/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.bin',
        'data/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml')
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # video capture object
    # cap = Sr.StreamReader(config.address_cam, 'Camera Car')
    # cap = cv2.VideoCapture("rtsp://admin:tipa1234@192.168.88.243:554/cam/realmonitor?channel=1&subtype=0")
    cap = cv2.VideoCapture(config.address_cam)
    # load images
    camera_fail = cv2.imread('images/background_falha.jpg')

    # Show image configuration
    cv2.namedWindow('Zion', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Zion', 30, 0)

    # CarDetection
    # car_detector = Cd.CarDetection()

    # print("Main process running! pid:", os.getpid())
    # time_start = timeit.default_timer()

    while True:
        # Capture the video frame
        _, frame_to_process = cap.read()

        if frame_to_process is None:
            show_camera_fail(camera_fail)
            continue

        # if not car_detector.found_car():
        #     print('Carro Não encontrado')
        # else:
        #     print('Carro encontrado!')
        # position_car = car_detector.CarDetection()

        car_coord, point_center = network_infer(frame_to_process, net)
        if car_coord != -1:
            # print('cars: ', cars[0], cars[1], cars[2], cars[3])
            print('car_coord: ', car_coord)
            print('point_center: ', point_center)
            cv2.rectangle(frame_to_process,
                          (car_coord[0], car_coord[1]), (car_coord[2], car_coord[3]),
                          color_green,
                          rectThinkness)
            cv2.circle(frame_to_process, point_center, 3, color_red, rectThinkness*3)
            # plate = plateCar(frame_to_process)
            # print('plate: ', plate)

        # Exibe a data atual
        curr_date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
        cv2.putText(frame_to_process, 'Data: ' + curr_date, (15, 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (10, 255, 10), 1)

        cv2.imshow('Zion', frame_to_process)

        # the 'q' button is set as the quitting button
        if cv2.waitKey(5) & 0xFF == ord('q'):
            # ending the program
            cap.stop_thread()
            break

    print('Main process done.')
    return


if __name__ == "__main__":
    # main()
    try:
        main()
    except Exception as error:
        print('Main process crashed.')
        print(error)

    print("\nAll clear. \n")
    cv2.destroyAllWindows()
