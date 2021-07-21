import queue
import threading
import cv2


class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


def save_video():
    # cap_video = VideoCapture('rtsp://admin:g551nt3l@sunlake.letmein.com.br:569/cam/realmonitor?channel=1&subtype=0')
    # cap_video = VideoCapture('rtsp://admin:tipa1234@192.168.88.41:554/cam/realmonitor?channel=1&subtype=0')
    # cap_video = VideoCapture('rtsp://admin:g551nt3l@sunlake.letmein.com.br:570?channel=1&stream=0.sdp')
    cap_video = VideoCapture('rtsp://admin:tipa1234@saojoaquim.letmein.com.br:560/cam/realmonitor?channel=1&subtype=0')
    # cap_video = VideoCapture("rtsp://admin:tipa1234@192.168.88.243:554/cam/realmonitor?channel=1&subtype=0")
    frame = cap_video.read()
    frame_height, frame_width, _ = frame.shape
    print('frame_height:', frame_height)
    print('frame_width:', frame_width)
    out_video = cv2.VideoWriter('/home/alan/dataset_letmein/dataset_radar.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while True:
        frame = cap_video.read()#.copy()

        out_video.write(frame)

        cv2.imshow("frame-radar", frame)

        if chr(cv2.waitKey(80) & 255) == 'q':
            break


if __name__ == '__main__':
    save_video()
