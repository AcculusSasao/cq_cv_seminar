# CQ出版社 画像認識セミナー用 プログラム

## Avnet MaaXBoard 8ULP (NXP i.MX 8ULP) 環境

ファームウェア https://github.com/Avnet/MaaXBoard-8ULP-HUB<br>
* Yocto Linux Full Image (wic) : avnet-image-full-maaxboard-8ulp-20231024030111.rootfs.wic
* BootLoader u-boot Image : u-boot-maaxboard-8ulp.imx

各ソフトウェアバージョン
* Python3 3.11.2
* opencv-python 4.7.0
* tflite_runtime 2.14.0
* PySimpleGUIWeb 0.39.0
* (pyzbar 0.1.9)

## ライセンス

[The MIT License (MIT),  Copyright (c) 2024 Yukiyoshi Sasao](LICENSE.txt)<br>
<br>
ただし、各AIモデルと、ポスト処理の一部ソースコードは、外部の物を使用します。それらは参照元のライセンスに従ってください。<br>
多くのモデルを [ax株式会社](https://axinc.jp/) 公開のものを使用していますのでそちらも参照ください。<br>
https://github.com/axinc-ai/ailia-models-tflite<br>

## モデルの準備

モデルをダウンロードしていない場合は、初回にダウンロードが必要です。
```
./download.sh
```

各モデルの [netron](https://netron.app/) 分析画像は [netron/](netron/)にあります。

## 共通コマンドオプション

|引数|内容|
|:---:|:---|
|-web|指定すると SimpleGUIWeb に結果を表示します。ポートは 1234 です。<br>指定した場合はブラウザで http://<ボードのIPアドレス>:1234/ を閲覧してください。 |
|--no_display|指定すると結果の画像表示をせず、シェルにprintするのみです。<br>処理の時間計測をしたいだけの時等に使用します。|
|-i <画像ファイル>| 入力をカメラ0ではなく画像ファイルを用います。 |

## 0. cap : 基本となるUSBカメラキャプチャ

```
python3 0_cap.py [options]
```

## 1. blazeface : 顔検出

```
python3 1_blazeface.py [options]
```

モデルおよびソースコード blazeface_utils.py, anchors.npy は以下のサイトのものを使用します。<br>
これらのライセンスは以下サイトに従ってください。<br>
https://github.com/axinc-ai/ailia-models-tflite/tree/main/face_detection/blazeface<br>

## 2. QRコード検出 by OpenCV

```
python3 2_qrcode.py [options]
```

## 3. 2Dバーコード検出 by Zbar

zbar:<br>
https://zbar.sourceforge.net/

```
python3 3_zbar.py [options]
```

現状 MaaXBoard 8ULP では、ライブラリ libzbar0 が不足しているため実行できません。

## 4. movenet : 人姿勢検出

```
python3 4_humanpose.py [options]
```

モデルはGoogle社による以下のものを使用します。<br>
https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8/

## 5. efficientnet_lite : 画像識別

```
python3 5_efficientnet.py [options]
```

モデルおよびソースコード efficientnet_lite_labels.py は以下のサイトのものを使用します。<br>
これらのライセンスは以下サイトに従ってください。<br>
https://github.com/axinc-ai/ailia-models-tflite/tree/main/image_classification/efficientnet_lite

## 6. efficientdet_lite : 物体検出

```
python3 6_efficientdet.py [options]
```

モデルは以下のサイトのものを使用します。<br>
これらのライセンスは以下サイトに従ってください。<br>
https://github.com/axinc-ai/ailia-models-tflite/tree/main/object_detection/efficientdet_lite

## 7. blazehand : 手検出、手骨格推定

```
python3 7_blazehand.py [options]
```

モデルおよびソースコード blazehand_utils.py, anchors.npy は以下のサイトのものを使用します。<br>
これらのライセンスは以下サイトに従ってください。<br>
https://github.com/axinc-ai/ailia-models-tflite/tree/main/hand_recognition/blazehand

## 8. OpenCV 画像処理

* cv2.calcOpticalFlowFarneback による密なオプティカルフロー算出
* 4方向に分けて2値化
* cv2.findContours による輪郭検出

```
python3 8_flowcontour.py [options]
```
