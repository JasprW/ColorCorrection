# 色卡定位及照片色彩校正
## 依赖包
* opencv-python
* numpy

## find_card.py
### find_corner(img, b=2, debug=False)
获取色卡四角的定位点
:param img: 输入图像
:param b: Adaptive Threshold 参数
:param debug: 是否使用debug模式，输出各步骤结果和图片
:return: 四个角点坐标

### get_color_card(img, points)
通过角点提取色卡部分，并对色卡进行透视校正，返回校正后色卡正视图
:param img: 输入图像
:param points: 色卡角点坐标
:return: 透视校正后的色卡图片


## color_correction.py
### extract_color(color_card)
获取实际拍摄的照片中色卡各颜色色值
:param color_card: 透视校正完成的色卡图片
:return: 色卡中的颜色色值，以矩阵格式存储，color_matrix，shape: (3 , 24)

### recorrect_color(raw_img, A):
用系数矩阵A对图像进行颜色校正
:param raw_img: 原始图像
:param A: 系数矩阵
:return: 返回校正后的图像

## color_direction_detect.py
### is_upsideDown(color_card)
判断色卡是否旋转

### is_mirrored(color_card)
判断色卡是否镜像

### is_upsideDown_and_mirrorred(color_card)
判断色卡是否镜像+旋转


## Usage
### 提取单张照片中的色卡：
```
python3 find_card.py /WHERE/TO/FILE_PATH
```

### 批量提取文件夹中的色卡（色卡文件名为从1开始递增数字，后缀为.jpg）：
```
python3 find_card.py /WHERE/TO/DIR_PATH/
```

### 对单张照片校色
```
python3 color_correction.py /WHERE/TO/FILE_PATH
```
