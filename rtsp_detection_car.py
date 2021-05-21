import cv2
import timeit
import math
import imutils
import numpy as np
from imutils import paths

def drawText(frame, scale, rectX, rectY, rectColor, text):
    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)

    top = max(rectY - rectThinkness, textSize[0])

    cv2.putText(frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3)
def calculate_centr(coord):
    return (coord[0] + (coord[2] / 2), coord[1] + (coord[3] / 2))


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

if __name__ == '__main__':
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
    net = cv2.dnn.Net_readFromModelOptimizer(
        'data/vehicle-detection-0202/FP32/vehicle-detection-0202.xml',
        'data/vehicle-detection-0202/FP32/vehicle-detection-0202.bin')

    # Load the model detection-crossroad-1016 (Pessoas, veiculos e bicicletas) 512x512
    # net = cv2.dnn.Net_readFromModelOptimizer(
    #     'data/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml',
    #     'data/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.bin')


    # Specify target device
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    # http://sunlake.letmein.com.br:8110/
    # video = cv2.VideoCapture("rtsp://admin:tipa1234@192.168.90.244:554/cam/realmonitor?channel=1&subtype=0")
    # video = cv2.VideoCapture("rtsp://admin:tipa1234@saojoaquim.letmein.com.br:559/cam/realmonitor?channel=1&subtype=0")
    video = cv2.VideoCapture('rtsp://admin:g551nt3l@sunlake.letmein.com.br:569/cam/realmonitor?channel=1&subtype=0')
    # video = cv2.VideoCapture(0)
    print('abriu camera!')

    # -- Capture frame-by-frame
    _, first_frame = video.read()
    (rows, cols, channels) = first_frame.shape
    print("shape: ", first_frame.shape)  # 1080x1920x3

    color_font = (0, 0, 0)
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)
    radius = 2
    thickness = 4

    array_center_car = []
    while (True):
        ############## Time ##############
        start = timeit.default_timer()

        # -- Capture frame-by-frame
        _, frame = video.read()

        # -- Capture frame-by-frame
        # plateRecognition(frame)



        # name: "input" , shape: [1x3x300x300] - An input image in the format [BxCxHxW], where:
        # Chama primeira rede neural referente detecção do veículo
        # blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
        # blob = cv2.dnn.blobFromImage(frame, size=(384, 672), ddepth=cv2.CV_8U)
        blob = cv2.dnn.blobFromImage(frame, size=(512, 512), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        region_x = int(1080/2)
        region_y = int(1920/2)
        # Recorte do veiculo identificado
        cuttImg = frame[region_x:1080, region_y:1920]

        listDetections = []
        # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
        for detection in out.reshape(-1, 7):
            # image_id = detection[0]
            # label = detection[1]
            conf = detection[2]

            if conf < 0.95:
                continue

            listDetections.append(detection)

        print("listDetections: ", len(listDetections))

        for detection in listDetections:
            # print('conf: ', conf, ' image_id: ', image_id, ' label: ', label)
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            coord = xmin, xmax, ymin, ymax
            center = calculate_centr_cut(coord)
            # print('coord: ', coord)
            print('calculate_centr: ', center)

            if len(array_center_car) == 0:
                array_center_car = np.append(array_center_car, center)
                print("array_center_car 1: ", array_center_car[0])
            # elif center[0] == array_center_car[0] || center[1] == array_center_car[1]:
            #      print("entrou 2")
            #      array_center_car = np.append(array_center_car, center)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), vColor, rectThinkness)

            stop = timeit.default_timer()
            time_cascade = round((stop - start) * 1000, 1)
            # print('Time:', time_cascade, 'ms')
            ############## Time ##############
            print('#######################\n')
            # putText(frame, str(label), xmin, ymin, color_font)
            # Print do centro
            center = int(center[0]), int(center[1])
            frame = cv2.circle(frame, center, radius, color_red, thickness)
            # putTextPrecision(frame, conf, xmax, ymax, (ymax-ymin), color_red)

        print("array_center_car: ", array_center_car)
        # -- Print img gray
        if cuttImg is not None:
            cv2.imshow('cur', cuttImg)
            showImg = imutils.resize(frame, height=800)
            cv2.imshow("showImg", showImg)


        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()