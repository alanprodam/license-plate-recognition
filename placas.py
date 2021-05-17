import cv2

def predict(frame, net):
    # Prepare input blob and perform an inference
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()

    predictions = []

    # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
    # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]

    # Draw detected faces on the frame
    for detection in out.reshape(-1, 7):
        image_id, label, conf, x_min, y_min, x_max, y_max = detection

        if conf > 0.5:
            predictions.append(detection)
            print('conf: ', conf)

    # return the list of predictions to the calling function
    return predictions

# Load the model
lpr_net = cv2.dnn.Net_readFromModelOptimizer('data/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.xml', 
                                             'data/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.bin')

net = cv2.dnn.readNet('data/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml',
                      'data/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.bin')


# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)


# Read an image
frame = cv2.imread('images/carro2.jpeg')

predictions = predict(frame, net)

# Draw detected faces on the frame
for prediction in predictions:
    confidence = float(prediction[2])
    xmin = int(prediction[3] * frame.shape[1])
    ymin = int(prediction[4] * frame.shape[0])
    xmax = int(prediction[5] * frame.shape[1])
    ymax = int(prediction[6] * frame.shape[0])

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)


# scale_percent = 350 # percent of original size
# width = int(frame.shape[1] * scale_percent / 100)
# height = int(frame.shape[0] * scale_percent / 100)
# dim = (width, height)

# # resize image
# resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
# print('Resized Dimensions : ',resized.shape)

cv2.namedWindow('imgOpenvino', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('imgOpenvino', 0, 0)
# cv2.imshow('imgOpenvino', resized)
cv2.imshow('imgOpenvino', frame)
cv2.waitKey(0)

# Save the frame to an image file
#cv2.imwrite('out.png', frame)
