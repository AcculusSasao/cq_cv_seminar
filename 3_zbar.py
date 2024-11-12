# MIT License : Copyright (c) 2024 Yukiyoshi Sasao

# pip install pyzbar
# https://pypi.org/project/pyzbar/
from pyzbar.pyzbar import decode, ZBarSymbol
import sys
import cv2
import argparse

from webgui import PySimpleGUIWebWrapper
from ai_utils import TimeManager

if __name__ == '__main__':
    title = 'zbar'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default=None, help='input image file name')
    parser.add_argument('-web', '--web', action='store_true', help='show results on Web with port 1234')
    parser.add_argument('--cap_width', type=int, default=640, help='capture width')
    parser.add_argument('--cap_height', type=int, default=480, help='capture height')
    parser.add_argument('--no_display', action='store_true', help='no display and print results and time.')
    args = parser.parse_args()

    # list of bar types you want to detect
    bar_symbols = [
        ZBarSymbol.CODABAR,
        ZBarSymbol.CODE128,
        ZBarSymbol.CODE39,
        ZBarSymbol.CODE93,
        ZBarSymbol.EAN13,
        ZBarSymbol.EAN2,
        ZBarSymbol.EAN5,
        ZBarSymbol.EAN8,
        ZBarSymbol.ISBN10,
        ZBarSymbol.ISBN13,
        ZBarSymbol.UPCA,
        ZBarSymbol.UPCE,
    ]

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
        
        # detect Barcode
        decoded_list = decode(frame, symbols=bar_symbols)
        for decoded in decoded_list:
            x0 = decoded.rect.left
            y0 = decoded.rect.top
            x1 = x0 + decoded.rect.width
            y1 = y0 + decoded.rect.height
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 4)
            string = decoded.type + ': ' + decoded.data.decode()
            cv2.putText(frame, string, ((x0 + x1)//2, (y0 + y1)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # show result
        if args.no_display:
            print(tm.get_info())
            for decoded in decoded_list:
                print(decoded)
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
