# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import sys
import cv2
import argparse
import tflite_runtime.interpreter as tflite

from blazeface import blazeface_utils
from webgui import PySimpleGUIWebWrapper
from ai_utils import normalize_image, TimeManager

def draw_results(img, detections, offset):
    wh = max(img.shape[1], img.shape[0])
    for detection in detections:
        for d in detection:
            top_left = (int(d[1]*wh - offset[1]), int(d[0]*wh - offset[0]))
            bottom_right = (int(d[3]*wh - offset[1]), int(d[2]*wh - offset[0]))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 1)
            for k in range(6):
                kp_x = d[4 + k*2] * wh - offset[1]
                kp_y = d[4 + k*2 + 1] * wh - offset[0]
                cv2.circle(img, (int(kp_x), int(kp_y)), 2, (0, 0, 255), -1)

if __name__ == '__main__':
    title = 'blazeface'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default=None, help='input image file name')
    parser.add_argument('-web', '--web', action='store_true', help='show results on Web with port 1234')
    parser.add_argument('--cap_width', type=int, default=640, help='capture width')
    parser.add_argument('--cap_height', type=int, default=480, help='capture height')
    parser.add_argument('--no_display', action='store_true', help='no display and print results and time.')
    parser.add_argument('--model', type=str, default='blazeface/face_detection_front_128_full_integer_quant.tflite', help='path to model file')
    parser.add_argument('--num_threads', type=int, default=2, help='tflite runtime num_threads')
    parser.add_argument('--anchor', type=str, default='blazeface/anchors.npy', help='path to model file')
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
        img, scale, offset = normalize_image(frame, input_width, input_height, input_type, 127.5, 127.5, input_quantization)
        #print('normalize_image : img.shape={},dtype={}, scale={},offset={}'.format(img.shape, img.dtype, scale, offset))
        interpreter.set_tensor(input_index, img)
        
        # infer
        interpreter.invoke()
        
        # get output
        out0_classificators = interpreter.get_tensor(output_details[0]['index'])    # [1, x, 1]
        out1_regressors = interpreter.get_tensor(output_details[1]['index'])        # [1, x, 16]
        
        # post process
        detections = blazeface_utils.postprocess([out1_regressors, out0_classificators], anchor_path=args.anchor)
        draw_results(frame, detections, offset)
        
        # show result
        if args.no_display:
            print(tm.get_info())
            print('detections', detections)
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
