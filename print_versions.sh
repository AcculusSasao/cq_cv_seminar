#!/bin/sh

# 3.11.2
python3 --version

# 4.7.0
python3 -c "import cv2; print('cv2:', cv2.__version__)"

# 2.14.0
python3 -c "import tflite_runtime; print('tflite_runtime:', tflite_runtime.__version__)"

# '0.39.0  Released 6-Jun-2020'
python3 -c "import PySimpleGUIWeb; print('PySimpleGUIWeb:', PySimpleGUIWeb.version)"

# 0.1.9
python3 -c "import pyzbar; print('pyzbar:', pyzbar.__version__)"
