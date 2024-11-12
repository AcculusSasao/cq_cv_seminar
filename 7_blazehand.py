# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import sys
import cv2
import argparse
import tflite_runtime.interpreter as tflite
import numpy as np

from webgui import PySimpleGUIWebWrapper
from ai_utils import normalize_image, TimeManager
from blazehand import blazehand_utils as but

def decode_quant(val, quantization):
    val = val.astype(np.float32)
    sc, zp = quantization
    if sc != 0:
        val = (val - zp) * sc
    return val

def get_input_tensor(tensor, input_details, idx):
    details = input_details[idx]
    dtype = details['dtype']
    if dtype == np.uint8 or dtype == np.int8:
        quant_params = details['quantization_parameters']
        input_tensor = tensor / quant_params['scales'] + quant_params['zero_points']
        input_tensor = input_tensor.clip(0, 255)
        return input_tensor.astype(dtype)
    else:
        return tensor

def draw_results(img, keypoints):
    joints = but.HAND_CONNECTIONS
    for joint in joints:
        x0, y0 = keypoints[joint[0]][:2]
        x1, y1 = keypoints[joint[1]][:2]
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), thickness=4)
    for kps in keypoints:
        x, y = kps[:2]
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), thickness=4)

if __name__ == '__main__':
    title = 'efficientdet'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default=None, help='input image file name')
    parser.add_argument('-web', '--web', action='store_true', help='show results on Web with port 1234')
    parser.add_argument('--cap_width', type=int, default=640, help='capture width')
    parser.add_argument('--cap_height', type=int, default=480, help='capture height')
    parser.add_argument('--no_display', action='store_true', help='no display and print results and time.')
    parser.add_argument('--model', type=str, default='blazehand/palm_detection_builtin_256_full_integer_quant.tflite', help='path to detector model file')
    parser.add_argument('--model2', type=str, default='blazehand/hand_landmark_new_256x256_full_integer_quant.tflite', help='path to landmark model file')
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

    # init tflite Interpreter 2
    interpreter2 = tflite.Interpreter(model_path=args.model2, num_threads=args.num_threads)
    interpreter2.allocate_tensors()
    input_details2 = interpreter2.get_input_details()
    output_details2 = interpreter2.get_output_details()
    input_shape2 = input_details2[0]['shape']
    input_index2 = input_details2[0]['index']
    input_type2 = input_details2[0]['dtype']
    input_quantization2 = input_details2[0]['quantization']
    input_width2 = input_shape2[2]
    input_height2 = input_shape2[1]

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
        img, scale, offset = normalize_image(frame, input_width, input_height, input_type, 0, 255, input_quantization, b_swap_rb=False)
        interpreter.set_tensor(input_index, img)
        
        # infer
        interpreter.invoke()
        
        # get output
        preds_tf_lite = {}
        preds_tf_lite[0] = interpreter.get_tensor(output_details[1]['index'])   #1x2944x18 regressors
        preds_tf_lite[0] = decode_quant(preds_tf_lite[0], output_details[1]['quantization'])
        preds_tf_lite[1] = interpreter.get_tensor(output_details[0]['index'])   #1x2944x1 classificators
        preds_tf_lite[1] = decode_quant(preds_tf_lite[1], output_details[0]['quantization'])
        detections = but.detector_postprocess(preds_tf_lite, anchor_path='blazehand/anchors.npy')

        # landmark estimation
        if detections[0].size != 0:
            _scale = [1/scale, 1/scale]
            _pad = [int(offset[0] * scale), int(offset[0] * scale), int(offset[1] * scale), int(offset[1] * scale)]
            imgs, affines, _ = but.estimator_preprocess(
                frame, detections, _scale, _pad
            )

            landmarks = np.zeros((0,63))
            flags = np.zeros((0,1,1,1))
            handedness = np.zeros((0,1,1,1))

            for img_id in range(len(imgs)):
                est_input = get_input_tensor(np.expand_dims(imgs[img_id],axis=0), input_details2, 0)
                interpreter2.set_tensor(input_details2[0]['index'], est_input)
                interpreter2.invoke()

                _landmarks = interpreter2.get_tensor(output_details2[2]['index'])
                _landmarks = decode_quant(_landmarks, output_details2[2]['quantization'])
                landmarks = np.concatenate([landmarks, _landmarks], 0)
                _flags = interpreter2.get_tensor(output_details2[0]['index'])
                _flags = decode_quant(_flags, output_details2[0]['quantization'])
                flags = np.concatenate([flags, _flags], 0)
                _handedness = interpreter2.get_tensor(output_details2[1]['index'])
                _handedness = decode_quant(_handedness, output_details2[1]['quantization'])
                handedness = np.concatenate([handedness, _handedness], 0)

            normalized_landmarks = landmarks.reshape((landmarks.shape[0], -1, 3))
            normalized_landmarks = normalized_landmarks / 256.0
            flags = flags.squeeze((1, 2, 3))
            handedness = handedness.squeeze((1, 2, 3))

            landmarks = but.denormalize_landmarks(
                normalized_landmarks, affines
            )
            threshold = 0.30
            for i in range(len(flags)):
                landmark, flag, handed = landmarks[i], flags[i], handedness[i]
                #print(flag)
                if flag >= threshold:
                    draw_results(frame, landmark)

        # show result
        if args.no_display:
            print(tm.get_info())
            print(landmarks)
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
