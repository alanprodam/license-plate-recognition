import threading
import queue
import cv2
import time
import os


class StreamReader:
    def __init__(self, address, cam_name):
        self.address = address

        if self.address == '0' or self.address == 0:
            print('WARNING: Ignoring', cam_name)
            return
        self.cap = cv2.VideoCapture(self.address)
        self.q = queue.Queue()
        self.stop = False
        t = threading.Thread(target=self._reader)
        t.setDaemon(True)
        t.start()
        self.wait = False
        if 'rtsp' not in self.address:
            self.wait = True

    def put_frame(self, frame):
        if not self.q.empty():
            try:
                self.q.get_nowait()  # discard previous (unprocessed) frame
            except queue.Empty:
                pass
        self.q.put(frame)
        if self.wait:
            time.sleep(0.095)
    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):

        print("StreamReader running! pid: ", os.getpid())
        while True:
            if self.stop:
                print("StreamReader stopping...")
                time.sleep(0.02)
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        print("lost frame")
                        pass
                self.cap.release()
                break

            ret, frame = self.cap.read()
            if not ret:  # lost connection, must retry
                print("StreamReader: failed to read frame, address:", self.address)
                self.put_frame(None)
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.address)
                continue

            self.put_frame(frame)


        print("StreamReader done.")
        return

    def read(self):
        if self.address == '0' or self.address == 0:
            return -1
        else:
            return self.q.get()

    def stop_thread(self):
        self.stop = True
