# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import sys
import cv2
import argparse

from webgui import PySimpleGUIWebWrapper
from ai_utils import TimeManager

if __name__ == '__main__':
    title = 'qrcode'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default=None, help='input image file name')
    parser.add_argument('-web', '--web', action='store_true', help='show results on Web with port 1234')
    parser.add_argument('--cap_width', type=int, default=640, help='capture width')
    parser.add_argument('--cap_height', type=int, default=480, help='capture height')
    parser.add_argument('--no_display', action='store_true', help='no display and print results and time.')
    args = parser.parse_args()

    # OpenCV QRCodeDetector
    # https://docs.opencv.org/4.7.0/de/dc3/classcv_1_1QRCodeDetector.html
    qr = cv2.QRCodeDetector()

    # prepare camera
    if args.image:
        frame_image = cv2.imread(args.image)
        if frame_image is None:
            print('fail to open', args.image)
            sys.exit(-1)
        cap = None
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit(-1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
    
    # prepare web output
    if args.web:
        web = PySimpleGUIWebWrapper(window_name_list=[title])
    
    # capture loop
    tm = TimeManager()
    while True:
        tm.measure()
        if args.web and web.loop() < 0:
            print('finish by PySimpleGUIWeb')
            break
        
        # capture image
        if args.image:
            frame = frame_image.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print('end of capture')
                break
        
        # detect QR
        retval, decoded_info, points, straight_qrcode = qr.detectAndDecodeMulti(frame)
        
        # draw results
        if retval:
            cv2.polylines(frame, points.astype(int), True, (0, 255, 0), 4)
            for string, p in zip(decoded_info, points):
                cv2.putText(frame, string, ((p[0] + p[2]) / 2).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # show result
        if args.no_display:
            print(tm.get_info())
            if retval:
                for string, p in zip(decoded_info, points):
                    print(string, p)
        else:
            tm.draw(frame)
            if args.web:
                web.imshow(title, frame)
            else:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1)
                if key == 27:   # ESC
                    break
    if args.web:
        web.close()
