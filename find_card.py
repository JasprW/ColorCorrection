# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import sys
# from color_detect import *


def SortPoint(points):
    """
    对四个定位点进行排序，排序后顺序分别为左上、右上、左下、右下
    :param points: 待排序的点集
    :return: 排序完成的点集
    """
    sp = sorted(points, key=lambda x: (int(x[1]), int(x[0])))
    if sp[0][0] > sp[1][0]:
        sp[0], sp[1] = sp[1], sp[0]

    if sp[2][0] > sp[3][0]:
        sp[2], sp[3] = sp[3], sp[2]

    return sp


def find_corner(img):
    """
    获取色卡四角的定位点
    :param img: 输入图像
    :return: 四个角点坐标
    """
    # src = cv2.imread("IMG_0793.jpg")
    # img = cv2.resize(img, (2000, 1500), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # image_show("binary", binary)

    blur = cv2.GaussianBlur(binary, (5, 5), 0)
    # edges = cv2.Canny(blur, 50, 150)
    edges = cv2.Canny(blur, 100, 300)
    # image_show("", edges)

    _, contours, hierarchy = cv2.findContours(
        edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    found = []

    img_test = img.copy()

    for i in range(len(contours)):
        k = i
        c = 0

        # 获取轮廓包围框及其长宽
        rect = cv2.minAreaRect(contours[i])
        w = rect[1][0]
        h = rect[1][1]
        if w and h:
            rate = min(w, h) / max(w, h)
            # 选取方形轮廓
            if (rate >= 0.75 and w < img.shape[1] / 4 and h < img.shape[0] / 4):

                # cv2.drawContours(img_test, contours, i,
                #                  (255, 255, 0), 2, cv2.LINE_AA)

                # 判断轮廓层级，筛选多层轮廓的外围轮廓
                while hierarchy[k][2] != -1:
                    k = hierarchy[k][2]
                    c = c + 1
                    # 超过4层则判断为定位点
                    if c == 3:
                        cv2.drawContours(img_test, contours, i,
                                         (255, 0, 0), 5, cv2.LINE_AA)
                    if c >= 4:
                        cv2.drawContours(img_test, contours, i,
                                         (0, 0, 255), 5, cv2.LINE_AA)
                        found.append(i)
                        continue
        else:
            continue

    # image_show("test", img_test)

    temp_contours = []
    for i in found:
        temp_contours.append(contours[i])

    img_dc = img.copy()

    # 按轮廓面积从大到小排序
    contours = sorted(
        temp_contours, key=cv2.contourArea, reverse=True)[0:12]

    candidate_contours = []
    candidate_contours.append(contours[0])

    for i in range(1, 12):
        if is_duplicate(contours[i], candidate_contours):
            continue
        else:
            candidate_contours.append(contours[i])

    # print(len(candidate_contours))

    # 选取第2~5个轮廓作为定位点（最大的为图像外围轮廓）
    if len(candidate_contours) >= 4:
        candidate_contours = candidate_contours[0:4]
        # print(contours)
        for i in range(4):
            cv2.drawContours(img_dc, candidate_contours, i,
                             (0, 0, 255), 2, cv2.LINE_AA)
        # image_show("positioning", img_dc)
        location_points = []

        for i in range(0, 4):
            pos_rect = cv2.minAreaRect(candidate_contours[i])
            location_points.append(pos_rect[0])

        # 对定位点排序，排序后顺序为：左上，右上，左下，右下
        location_points = SortPoint(location_points)
        return location_points
    else:
        print(len(contours))
        for i in range(len(contours)):
            cv2.drawContours(img_dc, contours, i,
                             (255, 255, 0), 2, cv2.LINE_AA)
        # image_show("", img_dc)


def is_duplicate(c, contours):
    """
    判断当前轮廓是否与其他轮廓属于同一定位点，用于筛选同一定位点的唯一轮廓
    :param c: 当前判断的contour
    :param contours: 当前所有非重叠contours集合
    :return: contours中是否存在与c相交的轮廓，bool，重复轮廓返回True，不重复返回False
    """
    r = cv2.boundingRect(c)
    for contour in contours:
        rect = cv2.boundingRect(contour)
        if _intersection(r, rect):
            return True
        else:
            continue
    return False


def _intersection(a, b):
    """
    判断两个rect是否相交
    :param a: rect1
    :param b: rect2
    :return: 是否相交，bool， 相交为True，不相交为False
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False  # or (0, 0, 0, 0) ?
    return True


def get_color_card(img, points):
    """
    通过角点提取色卡部分，并对色卡进行透视校正，返回校正后色卡正视图
    :param img: 输入图像
    :param points: 色卡角点坐标
    :return: 透视校正后的色卡图片
    """
    # 对定位点排序，排序后顺序为：左上，右上，左下，右下
    sp = SortPoint(points)

    # 查看定位点位置及顺序
    img_point_position = img.copy()
    for i in range(0, 4):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img_point_position, (np.float32(sp[i][0]), np.float32(
            sp[i][1])), 20, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(img_point_position, str(
            i + 1), (np.int0(sp[i][0]) - 10, np.int0(sp[i][1]) + 10), font, 10, (255, 255, 0), 10)
    # image_show("Points position", img_point_position)

    # 透视变换，转换为正视图
    pts1 = np.float32(sp)
    pts2 = np.float32([[0, 0], [1000, 0], [0, 750], [1000, 750]])
    transform = cv2.getPerspectiveTransform(pts1, pts2)
    warpedimg = cv2.warpPerspective(img, transform, (1000, 750))
    # image_show("color card", warpedimg)

    # 设定裁剪边距，完全去除定位标志
    padding = np.int0(warpedimg.shape[0] * 0.06)
    img_cropped = warpedimg[padding:(
        warpedimg.shape[0] - padding), padding:(warpedimg.shape[1] - padding)]
    img_output = cv2.resize(img_cropped, (400, 300),
                            interpolation=cv2.INTER_CUBIC)
    # image_show("cropeped", img_cropped)

    return img_output


if __name__ == '__main__':
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_ext = os.path.splitext(os.path.basename(file_path))[1]
            dir_name = os.path.dirname(file_path)
        else:
            print ("未找到文件")
    else:
        print("参数数量错误")

    # img = cv2.imread('images/Image5.jpg', 1)
    img = cv2.imread(file_path)
    # img = cv2.imread('images/IMG_0793.jpg', 1)
    corner_points = find_corner(img)
    card = get_color_card(img, corner_points)
    cv2.imwrite(dir_name + '/' + file_name + '-card' + file_ext, card)
    # image_show('', card)
