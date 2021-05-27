import requests
import json
import cv2
from imutils import paths

url = 'https://lpr.letmein.com.br/upload'
color_red = (0, 0, 255)
rectThinkness = 2

# frameCropped = cv2.imread('testImgs/placa-de-carro_test.jpg')
# frameCropped = cv2.imread('testImgs/bra3r52.jpg')
# frameCropped = cv2.imread('testImgs/carro_placa.jpg')

def plateCar(frameCropped):
    _, imencoded = cv2.imencode(".jpg", frameCropped)
    file = {'file': ('lpr.jpg', imencoded)}
    r = requests.post(url, files=file)
    data = json.loads(r.text)

    print(data)
    print(data[0]['plate'])
    print(data[0]['possiblePlate'])
    print(data[0]['probabilities'])
    print(data[0]['width'])
    print(data[0]['x'])
    print(data[0]['y'])

    xmin = data[0]['x'] - data[0]['width']
    ymin = data[0]['y'] - data[0]['width']
    xmax = data[0]['x'] + data[0]['width']
    ymax = data[0]['y'] + data[0]['width']

    cv2.rectangle(frameCropped, (xmin, ymin), (xmax, ymax), color_red, rectThinkness)
    cv2.imshow('image', frameCropped)
    cv2.waitKey(0)

TEST_PATH = 'testImgs'

for imagePath in paths.list_images(TEST_PATH):
    print(imagePath)
    img = cv2.imread(imagePath)
    if img is None:
        continue
    try:
        plateCar(img)
    except Exception as Err:
        print('Err', Err)


cv2.destroyAllWindows()
