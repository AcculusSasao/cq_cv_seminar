# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import sys
import cv2
import argparse
import numpy as np

from webgui import PySimpleGUIWebWrapper
from ai_utils import TimeManager

if __name__ == '__main__':
    title = 'flowcontour'
    title2 = 'flow'
    parser = argparse.ArgumentParser()
    parser.add_argument('-web', '--web', action='store_true', help='show results on Web with port 1234')
    parser.add_argument('-s', '--proc_scale', type=float, default=16, help='resize input image by 1/proc_scale')
    parser.add_argument('--mag_threshold', type=float, default=1, help='threshold of magnitude')
    parser.add_argument('--area_threshold', type=int, default=5, help='threshold of contour area')
    parser.add_argument('--cap_width', type=int, default=640, help='capture width')
    parser.add_argument('--cap_height', type=int, default=480, help='capture height')
    parser.add_argument('--no_display', action='store_true', help='no display and print results and time.')
    args = parser.parse_args()

    # prepare camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
    
    # prepare web output
    if args.web:
        web = PySimpleGUIWebWrapper(window_name_list=[title, title2])
    
    # capture loop
    tm = TimeManager()
    prev_img = None
    while True:
        tm.measure()
        if args.web and web.loop() < 0:
            print('finish by PySimpleGUIWeb')
            break
        
        # capture image
        ret, frame = cap.read()
        if not ret:
            print('end of capture')
            break
        
        # optical flow
        img = cv2.resize(frame, None, fx=1/args.proc_scale, fy=1/args.proc_scale)
        hsv = np.zeros_like(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_img is None:
            prev_img = img
            continue
        flow = cv2.calcOpticalFlowFarneback(prev_img, img, flow=None, 
                                            pyr_scale=0.5, levels=1, winsize=5, iterations=1, 
                                            poly_n=5, poly_sigma=1.2, flags=0)
        prev_img = img
        # to be magnitude and angle
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # to be HSV to show
        hsv[...,1] = 255
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if not args.no_display:
            if args.web:
                web.imshow(title2, rgb)
            else:
                cv2.imshow(title2, rgb)
        
        # create bin images of 4 directions
        binimgs = [np.zeros(img.shape[:2], dtype=np.uint8) for _ in range(4)]
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if mag[y, x] >= args.mag_threshold:
                    val = ang[y, x] + np.pi / 4
                    while val < 0:
                        val += np.pi * 2
                    while val >= np.pi * 2:
                        val -= np.pi * 2
                    bin = int(val / np.pi / 2 * 4)
                    binimgs[bin][y, x] = 1
        # find contours for each bin images
        bincolors = (
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
        )
        results = []
        for binidx, binimg in enumerate(binimgs):
            contours, _hierarchy = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dst_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= args.area_threshold:
                    dst_contours.append(contour * args.proc_scale)
            cv2.drawContours(frame, dst_contours, -1, color=bincolors[binidx], thickness=2)
            results.append(dst_contours)
        
        # show result
        if args.no_display:
            print(tm.get_info())
            print(results)
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
