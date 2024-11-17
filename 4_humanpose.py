# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import sys
import cv2
import argparse
import tflite_runtime.interpreter as tflite

from webgui import PySimpleGUIWebWrapper
from ai_utils import normalize_image, TimeManager, POSE17_JOINTS

def draw_results(img, keypoints, th=0.1, b_draw_numbers=True, 
                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
    joints = POSE17_JOINTS
    for joint in joints:
        if keypoints[joint[0]][2] < th or keypoints[joint[1]][2] < th:
            continue
        x0, y0 = keypoints[joint[0]][:2]
        x1, y1 = keypoints[joint[1]][:2]
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), thickness=4)
    for kps in keypoints:
        x, y, s = kps
        if s < th:
            continue
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), thickness=4)
    if b_draw_numbers:
        for kidx, kps in enumerate(keypoints):
            x, y, s = kps
            if s < th:
                continue
            d = 2
            cv2.putText(img, str(kidx), (int(x+d), int(y+d)), fontFace, fontScale, (200,200,200), thickness)
            cv2.putText(img, str(kidx), (int(x), int(y)), fontFace, fontScale, (20,20,20), thickness)

def judge_raise_hands_and_draw(img, keypoints, th=0.1,
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
    lrs = ('Left', 'Right')
    shoulder_indices = (5, 6)
    wrist_indices = (9, 10)
    # loop for left, right
    for lr, shoulder_index, wrist_index in zip(lrs, shoulder_indices, wrist_indices):
        if keypoints[shoulder_index][2] < th or keypoints[wrist_index][2] < th:
            continue
        shoulder_y = keypoints[shoulder_index][1]
        wrist_y = keypoints[wrist_index][1]
        if shoulder_y > wrist_y:
            msg = lr + ' hand raised !'
            print(msg)
            x, y = keypoints[wrist_index][:2]
            cv2.putText(img, msg, (int(x), int(y)), fontFace, fontScale, (0,255,255), thickness)

if __name__ == '__main__':
    title = 'humanpose'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default=None, help='input image file name')
    parser.add_argument('-web', '--web', action='store_true', help='show results on Web with port 1234')
    parser.add_argument('--cap_width', type=int, default=640, help='capture width')
    parser.add_argument('--cap_height', type=int, default=480, help='capture height')
    parser.add_argument('--no_display', action='store_true', help='no display and print results and time.')
    parser.add_argument('--model', type=str, default='movenet/singlepose-lightning-tflite-int8.tflite', help='path to model file')
    parser.add_argument('--num_threads', type=int, default=2, help='tflite runtime num_threads')
    args = parser.parse_args()

    # init tflite Interpreter
    interpreter = tflite.Interpreter(model_path=args.model, num_threads=args.num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_index = input_details[0]['index']
    input_type = input_details[0]['dtype']
    input_quantization = input_details[0]['quantization']
    input_width = input_shape[2]
    input_height = input_shape[1]

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
        
        # pre process
        img, scale, offset = normalize_image(frame, input_width, input_height, input_type, 0, 1, input_quantization)
        interpreter.set_tensor(input_index, img)
        
        # infer
        interpreter.invoke()
        
        # get output
        out = interpreter.get_tensor(output_details[0]['index'])[0][0]
        # post process
        keypoints = []
        for kps in out:
            y, x, s = kps
            x = x * input_shape[2] / scale - offset[1]
            y = y * input_shape[1] / scale - offset[0]
            keypoints.append([x, y, s])
        draw_results(frame, keypoints)
        judge_raise_hands_and_draw(frame, keypoints)
        
        # show result
        tm.draw(frame)
        if args.no_display:
            print(tm.get_info())
            print('keypoints', keypoints)
        else:
            if args.web:
                web.imshow(title, frame)
            else:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1)
                if key == 27:   # ESC
                    break
    if args.web:
        web.close()
