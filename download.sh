#!/bin/sh

if [ "$#" -eq 1 ]; then
	echo clean models..
	rm -rf blazeface movenet efficientnetlite efficientdetlite blazehand
	exit
fi

_WGET="wget -N"
# remove -N if not supported.
#_WGET=wget

# blazeface
#   https://github.com/axinc-ai/ailia-models-tflite/tree/main/face_detection/blazeface
DST=blazeface
mkdir -p $DST
$_WGET -P $DST https://storage.googleapis.com/ailia-models-tflite/blazeface/face_detection_front_128_full_integer_quant.tflite
$_WGET -P $DST https://raw.githubusercontent.com/axinc-ai/ailia-models-tflite/refs/heads/main/face_detection/blazeface/blazeface_utils.py
$_WGET -P $DST https://github.com/axinc-ai/ailia-models-tflite/raw/refs/heads/main/face_detection/blazeface/anchors.npy

# movenet
#   https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8/
DST=movenet
mkdir -p $DST
curl -L -o model.tar.gz https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-lightning-tflite-int8/1/download
tar xf model.tar.gz
mv 4.tflite $DST/singlepose-lightning-tflite-int8.tflite
rm model.tar.gz

# efficientnetlite
#   https://github.com/axinc-ai/ailia-models-tflite/tree/main/image_classification/efficientnet_lite
DST=efficientnetlite
mkdir -p $DST
$_WGET -P $DST https://storage.googleapis.com/ailia-models-tflite/efficientnet_lite/efficientnetliteb0_quant_recalib.tflite
$_WGET -P $DST https://raw.githubusercontent.com/axinc-ai/ailia-models-tflite/refs/heads/main/image_classification/efficientnet_lite/efficientnet_lite_labels.py

# efficientdetlite
#   https://github.com/axinc-ai/ailia-models-tflite/tree/main/object_detection/efficientdet_lite
DST=efficientdetlite
mkdir -p $DST
$_WGET -P $DST https://storage.googleapis.com/ailia-models-tflite/efficientdet_lite/efficientdet_lite0_integer_quant.tflite

# blazehand
#   https://github.com/axinc-ai/ailia-models-tflite/tree/main/hand_recognition/blazehand
DST=blazehand
mkdir -p $DST
$_WGET -P $DST https://storage.googleapis.com/ailia-models-tflite/blazepalm/palm_detection_builtin_256_full_integer_quant.tflite
$_WGET -P $DST https://storage.googleapis.com/ailia-models-tflite/blazehand/hand_landmark_new_256x256_full_integer_quant.tflite
$_WGET -P $DST https://raw.githubusercontent.com/axinc-ai/ailia-models-tflite/refs/heads/main/hand_recognition/blazehand/blazehand_utils.py
$_WGET -P $DST https://github.com/axinc-ai/ailia-models-tflite/raw/refs/heads/main/hand_recognition/blazehand/anchors.npy
