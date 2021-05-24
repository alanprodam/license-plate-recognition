import cv2
import timeit
import math
import imutils
import numpy as np


# def drawText(frame, scale, rectX, rectY, rectColor, text):
#     textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
#
#     top = max(rectY - rectThinkness, textSize[0])
#
#     cv2.putText(frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3)
# def calculate_centr(coord):
#     return (coord[0] + (coord[2] / 2), coord[1] + (coord[3] / 2))


def calculate_centr_cut(coord):
    return (int(abs(coord[0] - coord[1]) / 2 + coord[0]), int(abs(coord[2] - coord[3]) / 2 + coord[2]))


def calculate_centr_distances(centroid_1, centroid_2):
    return math.sqrt((centroid_2[0] - centroid_1[0]) ** 2 + (centroid_2[1] - centroid_1[1]) ** 2)


def putText(frame, result, x1, y1, color_font):
    cv2.putText(frame,
                result,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_font, 2)  # Nome da clase detectada


def putTextPrecision(frame, conf, x2, y2, box_h, color_font):
    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color_font, 2)  # Certeza de precisão da classe


def segmentationOpenCV(video):
    MIN_EDGE = 450
    MAX_EDGE = 500
    while True:
        ############## Time ##############
        start = timeit.default_timer()

        # -- Capture frame-by-frame
        _, frame = video.read()

        if frame is None:
            continue

        resize_width = int(frame_width * 0.4)
        resize_height = int(frame_height*0.4)

        # -- Resize image with INTER_CUBIC
        resize = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

        # -- Convert in gray scale
        gray = cv2.cvtColor(frame,
                            cv2.COLOR_BGR2GRAY)  # -- remember, OpenCV stores color images in Blue, Green, Red

        # -- Detection de edges
        edges = cv2.Canny(gray, MIN_EDGE, MAX_EDGE, apertureSize=3, L2gradient=True)  # default (350,400)

        # -- Blur bilateral filter
        blur = cv2.bilateralFilter(edges, 3, 75, 75)

        numLines = 4

        # Deteccao de linhas
        lines = cv2.HoughLines(blur, numLines, np.pi / 90, 100)

        if lines is not None:
            if lines.shape[0] >= numLines:
                x = 0
                med_theta = 0
                for i in range(0, numLines):
                    for rho, theta in lines[i]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        cv2.line(resize, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # med_theta = med_theta + (theta / numLines)
                        # lines_vector[i] = theta
                        x = x + x1 + x2

        stop = timeit.default_timer()
        time_cascade = round((stop - start) * 1000, 1)
        print('Time:', time_cascade, 'ms')

        # showImg = imutils.resize(resize, height=800)
        cv2.imshow("showImg", blur)
        cv2.imshow("showImg2", frame)

        if cv2.waitKey(106) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def segmentationOpenVino(video):
    # Load the model vehicle-recognition-0039 (Identifica o tipo de carro)
    net = cv2.dnn.Net_readFromModelOptimizer(
        'data/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml',
        'data/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.bin')

    # Specify target device
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    while True:
        ############## Time ##############
        start = timeit.default_timer()

        # -- Capture frame-by-frame
        _, frame = video.read()

        if frame is None:
            continue

        blob = cv2.dnn.blobFromImage(frame, size=(512, 896), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        # print('out: ', out)

        listDetections = []
        # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
        for detection in out.reshape(-1, 4):
            # image_id = detection[0]
            # label = detection[1]
            conf = detection[1]

            if conf < 0.5:
                continue

            print('conf: ', conf)
            listDetections.append(detection)

        print("listDetections: ", len(listDetections))

        for detection in listDetections:
            # print('conf: ', conf, ' image_id: ', image_id, ' label: ', label)
            xmin = int(detection[2] * frame.shape[1])
            ymin = int(detection[3] * frame.shape[0])
            # xmax = int(detection[5] * frame.shape[1])
            # ymax = int(detection[6] * frame.shape[0])
        #
        #
            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), vColor, rectThinkness)
            stop = timeit.default_timer()
            time_cascade = round((stop - start) * 1000, 1)
            print('Time:', time_cascade, 'ms')
        #     ############## Time ##############
        #     putText(frame, str(label), xmin, ymin, color_font)



        showImg = imutils.resize(frame, height=800)
        cv2.imshow("showImg", showImg)


        if cv2.waitKey(106) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def getRadar(video):
    vColor = (0, 255, 0)  # vehicle bounding-rect and information color
    pColor = (0, 0, 255)  # plate bounding-rect and information color
    rectThinkness = 2

    # Load the model vehicle-recognition-0039 (Identifica o tipo de carro)
    # net = cv2.dnn.Net_readFromModelOptimizer(
    #     'data/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.xml',
    #     'data/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.bin')

    # Load the model license-0106 (Carro e placa frontal)
    # net = cv2.dnn.Net_readFromModelOptimizer(
    #     'data/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml',
    #     'data/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin')

    # Load the model vehicle-detection-0200 (Identifica veiculos) 300x300
    # net = cv2.dnn.Net_readFromModelOptimizer(
    #     'data/vehicle-detection-0200/FP32/vehicle-detection-0200.xml',
    #     'data/vehicle-detection-0200/FP32/vehicle-detection-0200.bin')

    # Load the model vehicle-detection-0201 (Identifica veiculos) 384x672
    # net = cv2.dnn.Net_readFromModelOptimizer(
    #     'data/vehicle-detection-0201/FP32/vehicle-detection-0201.xml',
    #     'data/vehicle-detection-0201/FP32/vehicle-detection-0201.bin')

    # # Load the model vehicle-detection-0202 (Identifica veiculos) 512x512
    # net = cv2.dnn.Net_readFromModelOptimizer(
    #     'data/vehicle-detection-0202/FP32/vehicle-detection-0202.xml',
    #     'data/vehicle-detection-0202/FP32/vehicle-detection-0202.bin')

    # Load the model detection-crossroad-1016 (Pessoas, veiculos e bicicletas) 512x512
    net = cv2.dnn.Net_readFromModelOptimizer(
        'data/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml',
        'data/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.bin')

    # Specify target device
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    color_font = (0, 0, 0)
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)
    radius = 2
    thickness = 5

    image_id = 0

    while True:
        ############## Time ##############
        start = timeit.default_timer()

        # -- Capture frame-by-frame
        _, frame = video.read()

        # -- Capture frame-by-frame
        # plateRecognition(frame)

        region_init_x = int(1080 / 1.5)
        region_init_y = int(1920 / 7)

        if frame is None:
            continue

        # Recorte do veiculo identificado
        cutImgInit = frame[region_init_x:frame_width, region_init_y:int(region_init_y * 5)]
        coord_imgInit = region_init_x, frame_width, region_init_y, int(region_init_y * 5)

        # name: "input" , shape: [1x3x300x300] - An input image in the format [BxCxHxW], where:
        # Chama primeira rede neural referente detecção do veículo
        # blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
        # blob = cv2.dnn.blobFromImage(frame, size=(384, 672), ddepth=cv2.CV_8U)
        blob = cv2.dnn.blobFromImage(frame, size=(512, 512), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        listDetections = []
        array_center_car = []
        # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
        for detection in out.reshape(-1, 7):
            # image_id = detection[0]
            # label = detection[1]
            conf = detection[2]

            if conf < 0.3:
                continue

            print('conf: ', conf)
            listDetections.append(detection)

        print("listDetections: ", len(listDetections))

        for detection in listDetections:
            # print('conf: ', conf, ' image_id: ', image_id, ' label: ', label)
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            if len(array_center_car) == 0:
                coord = xmin, xmax, ymin, ymax
                point = calculate_centr_cut(coord)
                center = point, image_id
                # print('coord: ', coord)
                print('calculate_centr: ', center)
                array_center_car = np.append(array_center_car, center)
                print("array_center_car 1: ", array_center_car[0])
                # image_id += 1

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), vColor, rectThinkness)
                stop = timeit.default_timer()
                time_cascade = round((stop - start) * 1000, 1)
                print('Time:', time_cascade, 'ms')
                ############## Time ##############
                # putText(frame, str(label), xmin, ymin, color_font)
                # Print do centro

                frame = cv2.circle(frame, point, radius, color_red, thickness)
                # putTextPrecision(frame, conf, xmax, ymax, (ymax-ymin), color_red)

        print('#######################\n')
        print("array_center_car: ", array_center_car)
        showImg = imutils.resize(frame, height=800)
        cv2.imshow("showImg", showImg)
        # -- Print img cutImgInit
        if cutImgInit is not None:
            cv2.imshow('cutImgInit', cutImgInit)

        if cv2.waitKey(106) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # http://sunlake.letmein.com.br:8110/
    # video = cv2.VideoCapture("rtsp://admin:tipa1234@192.168.90.244:554/cam/realmonitor?channel=1&subtype=0")
    # video = cv2.VideoCapture("rtsp://admin:tipa1234@saojoaquim.letmein.com.br:559/cam/realmonitor?channel=1&subtype=0")
    # video = cv2.VideoCapture('rtsp://admin:g551nt3l@sunlake.letmein.com.br:569/cam/realmonitor?channel=1&subtype=0')
    video = cv2.VideoCapture('/home/alan/dataset_letmein/radar/dataset_radar.avi')
    # video = cv2.VideoCapture(0)

    print('abriu camera!')
    # -- Capture frame-by-frame
    _, first_frame = video.read()
    (frame_height, frame_width, channels) = first_frame.shape
    print("shape: ", first_frame.shape)  # 1080x1920x3

    # getRadar(video)
    segmentationOpenCV(video)
    # segmentationOpenVino(video)

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()
