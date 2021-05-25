import cv2
import timeit
import math
import imutils
import numpy as np

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
        # calculate_centr_distances(centroid_1, centroid_2)

        listDetections = []
        # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
        # A saída é um blob com a forma [B, C = 4, H = 512, W = 896]. Ele pode ser tratado como um mapa de recursos
        # de quatro canais, onde cada canal é uma probabilidade de uma das classes: BG, estrada, meio-fio, marca.
        for detection in out.reshape(-1, 4):
            # image_id = detection[0]
            # label = detection[1]
            channels = detection[1]

            print('channels: ', channels, 'type-channels: ', type(channels))

            stop = timeit.default_timer()
            time_cascade = round((stop - start) * 1000, 1)
            print('Time:', time_cascade, 'ms')

        showImg = imutils.resize(frame, height=800)
        cv2.imshow("showImg", showImg)

        if cv2.waitKey(106) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def velocityRadar(video, net):
    vColor = (0, 255, 0)  # vehicle bounding-rect and information color
    # pColor = (0, 0, 255)  # plate bounding-rect and information color
    rectThinkness = 2

    color_DarkSlateBlue = (72, 61, 139)
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)
    radius = 2
    thickness = 5

    image_id = 0
    center_last = 0
    car_tracked = False
    start = 0
    listPosition = []
    array_center_car = []

    while True:
        # -- Capture frame-by-frame
        _, frame = video.read()

        if frame is None:
            continue

        # espaço de calculo
        delta_s = 35

        # # Recorte do veiculo identificado
        frame_out = frame[550:frame_height, 200:frame_width - 800]
        cv2.rectangle(frame, (200, 550), (frame_width-800, frame_height), vColor, rectThinkness)

        # name: "input" , shape: [1x3x300x300] - An input image in the format [BxCxHxW], where:
        # Chama primeira rede neural referente detecção do veículo
        # blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
        # blob = cv2.dnn.blobFromImage(frame, size=(384, 672), ddepth=cv2.CV_8U)
        blob = cv2.dnn.blobFromImage(frame_out, size=(512, 512), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        listDetections = []
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
        for out_cars in out.reshape(-1, 7):
            classId = out_cars[1]
            conf = out_cars[2]
            if conf < 0.6 or classId != 1:
                continue

            xmin = int(out_cars[3] * frame_out.shape[1])
            ymin = int(out_cars[4] * frame_out.shape[0])
            xmax = int(out_cars[5] * frame_out.shape[1])
            ymax = int(out_cars[6] * frame_out.shape[0])
            cv2.rectangle(frame_out, (xmin, ymin), (xmax, ymax), color_red, rectThinkness)

            coord = xmin, xmax, ymin, ymax
            point_center = calculate_centr_cut(coord)

            print('point_center: ', point_center)
            detection = out_cars[3], out_cars[4], out_cars[5], out_cars[6]
            # print('conf: ', conf, ' classId: ', classId, 'image_id: ', image_id)
            # print('listPosition: ', listPosition, ' car_tracked: ', car_tracked)
            # image_id += 1

            listDetections.append(detection)

        print('-------------------------')
        print('listDetections: ', listDetections)
        print("size listDetections: ", len(listDetections))

        if len(listDetections) != 0:
            for image_id, detection in enumerate(listDetections):
                if not car_tracked:
                    print('Entrou primeira interacao...')


                    # array_center_car = np.append(array_center_car, point_center)
                    # listPosition.append(point_center)
                    # print("size listPosition: ", len(listPosition))
                    # center_last = point_center
                    # image_id += 1

                    # cv2.waitKey(0)

                # elif len(listPosition) != 0 and car_tracked == False and len(listDetections) == 1:
                #     print('Carro trackeado!!....')
                #     xmin = int(detection[3] * frame_out.shape[1])
                #     ymin = int(detection[4] * frame_out.shape[0])
                #     xmax = int(detection[5] * frame_out.shape[1])
                #     ymax = int(detection[6] * frame_out.shape[0])
                #
                #     coord = xmin, xmax, ymin, ymax
                #     point = calculate_centr_cut(coord)
                #     # center = point, image_id
                #     distancia = int(calculate_centr_distances(center_last, point))
                #     print('distancia: ', distancia, 'center_last: ', center_last, 'point: ', point)
                #     cv2.rectangle(frame_out, (xmin, ymin), (xmax, ymax), color_red, rectThinkness)
                #
                #     if distancia < 150:
                #         car_tracked = True
                #         array_center_car = np.append(array_center_car, point)
                #         listPosition.append(point)
                #         print('car_tracked: ', car_tracked, '| size_arry: ', len(listPosition))
                #         ############## Time ##############
                #         start = timeit.default_timer()
                #         print('car_tracked: ', car_tracked)
                #
                #         cv2.rectangle(frame_out, (xmin, ymin), (xmax, ymax), vColor, rectThinkness)
                #         cv2.line(frame_out, center_last, point, color_DarkSlateBlue, 3)
                #
                #         cv2.circle(frame_out, center_last, radius, color_red, thickness)
                #         cv2.circle(frame_out, point, radius, color_red, thickness)
                #
                #         center_last = point
                #     else:
                #         print('car_tracked falhou!')
                #         car_tracked = False
                #         listPosition = []
                #
                #     # cv2.waitKey(0)
                #
                # elif len(listPosition) != 0 and car_tracked == True and len(listDetections) == 1:
                #     print('Manutencao do tracking!!')
                #     xmin = int(detection[3] * frame_out.shape[1])
                #     ymin = int(detection[4] * frame_out.shape[0])
                #     xmax = int(detection[5] * frame_out.shape[1])
                #     ymax = int(detection[6] * frame_out.shape[0])
                #
                #     coord = xmin, xmax, ymin, ymax
                #     point = calculate_centr_cut(coord)
                #     # center = point, image_id
                #     distancia = int(calculate_centr_distances(center_last, point))
                #     print('distancia: ', distancia, 'center_last: ', center_last, 'point: ', point)
                #
                #     if distancia < 150:
                #         array_center_car = np.append(array_center_car, point)
                #         listPosition.append(point)
                #         print('car_tracked: ', car_tracked, '| size_arry: ', len(listPosition))
                #
                #         cv2.rectangle(frame_out, (xmin, ymin), (xmax, ymax), vColor, rectThinkness)
                #         cv2.line(frame_out, center_last, point, color_DarkSlateBlue, 3)
                #
                #         cv2.circle(frame_out, center_last, radius, color_red, thickness)
                #         cv2.circle(frame_out, point, radius, color_red, thickness)
                #
                #         center_last = point
                #     else:
                #         car_tracked = False
                #         listPosition = []
                #         stop = timeit.default_timer()
                #         time_cascade = round((stop - start) * 1000, 1)
                #         print('Time:', time_cascade, 'ms')
                #         print('car_tracked: ', car_tracked)

                    # cv2.waitKey(0)

        # elif len(listPosition) != 0 and car_tracked == True and len(listDetections) == 0:
        #     car_tracked = False
        #     stop = timeit.default_timer()
        #     listPosition = []
        #     average_time = round((stop - start) * 1, 1)
        #     print('Time:', average_time, 's')
        #     velocidade = round((float(delta_s) / float(average_time)) * 3.6)
        #     print('velocidade: ', velocidade, ' km/h')
        #     print('car_tracked: ', car_tracked)
        #
        #     cv2.waitKey(0)

        print('#######################\n')

        # first = True
        # last_point = 0
        # # print('listPosition: ', listPosition)
        # if len(listPosition) >= 2:
        #     for point in listPosition:
        #         if first:
        #             last_point = point
        #             # print('last_point: ', last_point)
        #             first = False
        #         else:
        #             current_point = point
        #             # print('current_point: ', current_point)
        #             cv2.line(frame_out, last_point, current_point, color_blue, 1)
        #             last_point = current_point

        # showImg = imutils.resize(frame_out, height=800)
        # cv2.imshow("frame_out", showImg)

        # -- Print img cutImgInit
        # if cutImgInit is not None:
        #     cv2.imshow('cutImgInit', cutImgInit)

        if frame is not None:
            showImgOut = imutils.resize(frame_out, height=800)
            cv2.imshow('frame_out', showImgOut)

            showImg = imutils.resize(frame, height=800)
            cv2.imshow("frame-radar", showImg)

        if cv2.waitKey(106) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # http://192.168.88.41/
    # video = cv2.VideoCapture("rtsp://admin:tipa1234@192.168.88.41:554/cam/realmonitor?channel=1&subtype=0")
    video = cv2.VideoCapture('/home/alan/dataset_letmein/radar_1920x1080/dataset_radar_4.avi')

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

    print('abriu camera!')
    # -- Capture frame-by-frame
    _, first_frame = video.read()
    (frame_height, frame_width, channels) = first_frame.shape
    print("shape: ", first_frame.shape)  # 1080x1920x3

    velocityRadar(video, net)

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()
