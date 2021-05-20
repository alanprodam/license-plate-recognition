import numpy as np
import cv2, timeit
import imutils
from imutils import paths

def drawText(frame, scale, rectX, rectY, rectColor, text):
    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)

    top = max(rectY - rectThinkness, textSize[0])

    cv2.putText(frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3)


def plateRecognition(frame):

    # name: "input" , shape: [1x3x300x300] - An input image in the format [BxCxHxW], where:
    # Chama primeira rede neural referente detecção do veículo
    pd_blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
    pd_net.setInput(pd_blob)
    out_pb = pd_net.forward()


    # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
    # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
    # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
    for detection in out_pb.reshape(-1, 7):
        conf = detection[2]
        # print('conf: ', conf)
        if conf < 0.1:
            continue
        image_id, label, conf, x_min, y_min, x_max, y_max = detection


        # Classe de identificação do carro
        classId = int(detection[1])
        if classId == 1:  # car
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            rectW = xmax - xmin
            print("rectW:", rectW)
            if rectW < 72:  # Minimal weight in vehicle-attributes-recognition-barrier-0039  is 72
                continue
            try:
                # Recorte do veiculo identificado
                attrImg = frame[ymin:ymax + 1, xmin:xmax + 1]

                # Identificação do tipo de carro e cor
                attr_blob = cv2.dnn.blobFromImage(attrImg, size=(72, 72), ddepth=cv2.CV_8U)
                attr_net.setInput(attr_blob, 'input')

                out_color = attr_net.forward("color")
                out_type = attr_net.forward("type")

                carColor = "Color: " + CAR_COLORS[np.argmax(out_color.flatten())]
                carType = "Type: " + CAR_TYPES[np.argmax(out_type.flatten())]

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), vColor, rectThinkness)

                drawText(frame, rectW * 0.002, xmin, ymin, vColor, carColor + " " + carType)
            except:
                print("Erro! attr_blob = cv2.dnn.blobFromImage(attrImg, size=(72, 72), ddepth=cv2.CV_8U)")


        # Classe de identificação das placas
        elif classId == 2:  # plate
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            xmin = max(0, xmin - 5)
            ymin = max(0, ymin - 5)
            xmax = min(xmax + 5, frame.shape[1] - 1)
            ymax = min(ymax + 5, frame.shape[0]- 1)

            rectW = xmax - xmin
            if rectW < 94:  # Minimal weight in plate-recognition-barrier-0001 is 94
                continue

            # Recorte do veiculo identificado
            # Crop a license plate. Do some offsets to better fit a plate.
            lpImg = frame[ymin:ymax + 1, xmin:xmax + 1]
            blob = cv2.dnn.blobFromImage(lpImg, size=(94, 24), ddepth=cv2.CV_8U)
            lpr_net.setInput(blob, 'data')
            lpr_net.setInput(seq_ind, 'seq_ind')
            out_lpr = lpr_net.forward()

            content = ''
            for idx in np.int0(out_lpr.reshape(-1)):
                if idx == -1:
                    break
                content += items[idx]

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), pColor, rectThinkness)
            drawText(frame, rectW * 0.008, xmin, ymin, pColor, content)
            # cv2.imshow('imgOpenvino', frame)

            # showImg = imutils.resize(frame, height=600)
            # cv2.imshow("showImg", showImg)

if __name__ == '__main__':
    vColor = (0, 255, 0)  # vehicle bounding-rect and information color
    pColor = (0, 0, 255)  # plate bounding-rect and information color
    rectThinkness = 2

    CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
    CAR_TYPES = ["Carro", "Onibus", "Caminhao", "Van"]

    items = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
             "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
             "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
             "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
             "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
             "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
             "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
             "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
             "<Zhejiang>", "<police>",
             "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
             "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
             "U", "V", "W", "X", "Y", "Z"]

    seq_ind = np.ones([88, 1], dtype=np.float32)
    seq_ind[0, 0] = 0

    # Load the model
    # lpr_net = cv2.dnn.Net_readFromModelOptimizer(
    #     'data/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.xml',
    #     'data/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.bin')
    #
    # attr_net = cv2.dnn.Net_readFromModelOptimizer(
    #     'data/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.xml',
    #     'data/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039.bin')
    #
    # pd_net = cv2.dnn.readNet(
    #     'data/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml',
    #     'data/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin')
    #
    # yolo_net = cv2.dnn.readNet(
    #     'data/yolo-v2-tiny-vehicle-detection-0001/FP16/yolo-v2-tiny-vehicle-detection-0001.xml',
    #     'data/yolo-v2-tiny-vehicle-detection-0001/FP16/yolo-v2-tiny-vehicle-detection-0001.bin')

    net = cv2.dnn.readNet(
        'data/vehicle-detection-0200/FP16/vehicle-detection-0200.xml',
        'data/vehicle-detection-0200/FP16/vehicle-detection-0200.bin')

    # Specify target device
    # lpr_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # lpr_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    #
    # # Specify target device
    # attr_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # attr_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    #
    # # Specify target device
    # pd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # pd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    #
    # # Specify target device
    # yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    # Specify target device
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    video = cv2.VideoCapture("rtsp://admin:tipa1234@192.168.90.243:554/cam/realmonitor?channel=1&subtype=0")
    # video = cv2.VideoCapture("rtsp://admin:tipa1234@saojoaquim.letmein.com.br:559/cam/realmonitor?channel=1&subtype=0")
    # video = cv2.VideoCapture(0)
    print('abriu camera!')

    # -- Capture frame-by-frame
    _, first_frame = video.read()
    (rows, cols, channels) = first_frame.shape
    print("shape: ", first_frame.shape)  # 1080x1920x3

    while (True):
        ############## Time ##############
        start = timeit.default_timer()

        # -- Capture frame-by-frame
        _, frame = video.read()

        # -- Capture frame-by-frame
        # plateRecognition(frame)

        # name: "input" , shape: [1x3x300x300] - An input image in the format [BxCxHxW], where:
        # Chama primeira rede neural referente detecção do veículo
        blob = cv2.dnn.blobFromImage(frame, size=(256, 256), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()


        # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
        for detection in out.reshape(-1, 7):
            conf = detection[2]

            if conf < 0.95:
                continue

            # Classe de identificação do carro
            # classId = int(detection[1])
            # if classId == 1:  # car
            # if conf > 0.6:  # car
            print('conf: ', conf)
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), vColor, rectThinkness)

        # -- Print img gray
        cv2.imshow('RTSP', frame)


        stop = timeit.default_timer()
        time_cascade = round((stop - start) * 1000, 1)
        # print('Time:', time_cascade, 'ms')
        ############## Time ##############

        # print('#######################\n')

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()