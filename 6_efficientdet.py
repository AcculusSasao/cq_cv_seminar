# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import sys
import cv2
import argparse
import tflite_runtime.interpreter as tflite

from webgui import PySimpleGUIWebWrapper
from ai_utils import normalize_image, TimeManager, COCO_CATEGORY

def post_process(bboxes, confs, class_ids, input_shape, scale, offset, threshold=0.4):
    category = COCO_CATEGORY
    input_width = input_shape[2]
    input_height = input_shape[1]
    
    results = []
    for bbox, conf, class_id in zip(bboxes, confs, class_ids):
        if conf < threshold:
            continue
        if class_id >= len(category):
            continue
        # bbox: ymin xmin ymax xmax
        y0, x0, y1, x1 = bbox
        y0 = y0 * input_height / scale - offset[0]
        y1 = y1 * input_height / scale - offset[0]
        x0 = x0 * input_width / scale - offset[1]
        x1 = x1 * input_width / scale - offset[1]
        # [bbox, conf, class_id, category]
        res = []
        res.append((x0, y0, x1, y1))
        res.append(conf)
        res.append(class_id)
        res.append(category[class_id])
        results.append(res)
    return results

def draw_results(img, results):
    for result in results:
        bbox, conf, class_id, category = result
        p0 = (int(bbox[0]), int(bbox[1]))
        p1 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, p0, p1, (0, 255, 0), 2)
        string = '{}({}) conf={:.3f}'.format(category, class_id, conf)
        pos = (p0[0], (p0[1] + p1[1])//2)
        cv2.putText(img, string, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

if __name__ == '__main__':
    title = 'efficientdet'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default=None, help='input image file name')
    parser.add_argument('-web', '--web', action='store_true', help='show results on Web with port 1234')
    parser.add_argument('--cap_width', type=int, default=640, help='capture width')
    parser.add_argument('--cap_height', type=int, default=480, help='capture height')
    parser.add_argument('--no_display', action='store_true', help='no display and print results and time.')
    parser.add_argument('--model', type=str, default='efficientdetlite/efficientdet_lite0_integer_quant.tflite', help='path to model file')
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
        #print(img)
        interpreter.set_tensor(input_index, img)
        
        # infer
        interpreter.invoke()
        
        # get output
        bboxes = interpreter.get_tensor(output_details[0]['index'])[0]
        confs = interpreter.get_tensor(output_details[1]['index'])[0]
        class_ids = interpreter.get_tensor(output_details[2]['index'])[0]
        # post process
        results = post_process(bboxes, confs, class_ids, input_shape, scale, offset)
        draw_results(frame, results)
        
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
