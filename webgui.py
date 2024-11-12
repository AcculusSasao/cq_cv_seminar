# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
''' GUI on Web using PySimpleGUIWeb
  preparation:
    pip install pysimpleguiweb htmlparser
    pip install remi --upgrade
'''
import cv2
import PySimpleGUIWeb as sg
import numpy as np
import io
from typing import List, Any

class PySimpleGUIWebWrapper:
    def __init__(self, title: str = 'Demo Application', window_name_list: List[str] = ['-IMAGE-'], web_port: int = 1234) -> None:
        results_layout = [[sg.Image(filename='', key=key)] for key in window_name_list]
        layout = [
            [
                sg.Text(title, size=(40, 1), justification='left')
            ],
            [
                sg.Button('Quit', size=(10, 1), key='-EXIT-'), 
            ],
            [sg.Frame("Results", results_layout)],
        ]
        self.window = sg.Window(title, layout, web_port=web_port, web_start_browser=False)

    def close(self):
        self.window.close()

    def loop(self, timeout_msec: int = 1) -> int:
        event, _values = self.window.read(timeout=timeout_msec)
        if event == "-EXIT-" or event == sg.WIN_CLOSED:
            return -1
        return 0

    def imshow(self, key: str, img: Any) -> None:
        if type(img) is np.ndarray:
            # cv2 image
            data = cv2.imencode(".png", img)[1].tobytes()
        elif hasattr(img, 'savefig'):
            # matplotlib.pyplot
            buf = io.BytesIO()
            img.savefig(buf, format="png")
            data = buf.getvalue()
        else:
            print('unknown img type', type(img))
            return
        self.window[key].update(data=data)
