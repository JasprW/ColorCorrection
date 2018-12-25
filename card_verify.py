# @Author: Jaspr
# @Date:   2018-12-10, 15:29:33
# @Email:  wang@jaspr.me
# @Last modified by:   Jaspr
# @Last modified time: 2018-12-25, 13:37:38

import sys
import os
import cv2
# from color_correction import extract_color


def is_upsideDown(color_card):
    from find_card import extract_color
    real_color = extract_color(color_card)

    upper_left = real_color[0]
    upper_right = real_color[5]
    bottom_left = real_color[18]
    bottom_right = real_color[23]

    if upper_left[2] < bottom_left[2] < upper_right[2] and upper_left[0] < bottom_right[0] < upper_right[0]:
        return True
    else:
        return False


def is_mirrored(color_card):
    from find_card import extract_color
    real_color = extract_color(color_card)

    upper_left = real_color[0]
    upper_right = real_color[5]
    bottom_left = real_color[18]
    bottom_right = real_color[23]

    return bool(bottom_left[0] < upper_left[0] < bottom_right[0] and bottom_left[2] < upper_right[2] < bottom_right[2])


def is_upsideDown_and_mirrorred(color_card):
    from find_card import extract_color
    real_color = extract_color(color_card)

    upper_left = real_color[0]
    upper_right = real_color[5]
    bottom_left = real_color[18]
    bottom_right = real_color[23]

    if upper_left[0] > bottom_left[0] > upper_right[0] and upper_left[2] > bottom_right[2] > upper_right[2]:
        return True
    else:
        return False


def is_card_ok(color_card):
    # from find_card import image_show
    gray = cv2.cvtColor(color_card.copy(), cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(
    #     gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # 使用自适应二值化避免过曝影响定位点识别
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    blur = binary
    edges = cv2.Canny(blur, 50, 150)

    # image_show("", edges)

    _, contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(
        contours, key=cv2.contourArea, reverse=True)
    c = 0
    img_test = color_card.copy()
    for i in range(10):
        cv2.drawContours(img_test, contours, i,
                         (0, 0, 255), 3, cv2.LINE_AA)
        rect = cv2.boundingRect(contours[i])
        # print(rect)
        w = rect[2]
        h = rect[3]
        if cv2.contourArea(contours[i]) / (w * h) < 0.9:
            c += 1
    # image_show("", img_test)
    # print(c)
    if c > 4:
        return False
    else:
        return True


def rotate(img, angle=180, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
        # 传入文件夹路径
        if os.path.isfile(path):
            file_path = path
            if os.path.isfile(file_path):
                card = cv2.imread(file_path)
                if not is_card_ok(card):
                    print("卡片不正常！")
                    sys.exit()
                else:
                    print("卡片正常！")
