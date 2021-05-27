import cv2
import timeit, time
import InputStreamReader as Sr
import os
import datetime
import config
from configs import config_application
import CarDetection as Fd
from multiprocessing.managers import BaseManager

color_black = (0, 0, 0)
color_blue = (255, 13, 13)
color_red = (13, 13, 210)
color_green = (13, 210, 13)
color_white = (255, 255, 255)

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

def main():
    # video capture object
    cap = Sr.StreamReader(config.address_cam, 'Camera Car')

    # Show image configuration
    cv2.namedWindow('Zion', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Zion', 30, 0)

    prev_list = []
    prev_track_recog = []
    print("Main process running! pid:", os.getpid())
    time_start = timeit.default_timer()
    time_out_processes = 10

    while True:
        # Capture the video frame
        frame_to_process = cap.read()


        # Exibe a data atual
        curr_date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
        cv2.putText(frame_to_process, 'Data: ' + curr_date, (15, 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (10, 255, 10), 1)

        cv2.imshow('LetMeIn', frame_to_process)

        # the 'q' button is set as the quitting button
        if cv2.waitKey(5) & 0xFF == ord('q'):
            # ending the program
            cap.stop_thread()
            # face_detector.stop_process()
            # # wait to join the processes
            # face_detector.process.join()
            # face_recognizer.process.join()
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