# -*- coding:utf-8 -*-
import cv2
import numpy as np
from color_detect import image_show

"""
img = cv2.imread("pt.jpg")
img = cv2.resize(img, (1200,900), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
# image_show("gray", gray)

edges = cv2.Canny(gray, 50, 250, apertureSize=3)
# cv2.imwrite("canny.jpg", edges)
# image_show("edges", edges)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                        minLineLength=100, maxLineGap=25)
result1 = img.copy()
print(lines)
# for i in range(int(np.size(lines) / 4)):
for i in range(int(np.size(lines) / 4)):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(result1, (x1, y1), (x2, y2), (0, 0, 255), 2)
image_show("", result1)
"""


def CrossPoint(line1, line2):
    x0, y0, x1, y1 = line1[0]
    x2, y2, x3, y3 = line2[0]

    dx1 = x1 - x0
    dy1 = y1 - y0

    dx2 = x3 - x2
    dy2 = y3 - y2

    D1 = x1 * y0 - x0 * y1
    D2 = x3 * y2 - x2 * y3

    y = float(dy1 * D2 - D1 * dy2) / (dy1 * dx2 - dx1 * dy2)
    x = float(y * dx1 - D1) / dy1

    return (int(x), int(y))


def SortPoint(points):
    sp = sorted(points, key=lambda x: (int(x[1]), int(x[0])))
    if sp[0][0] > sp[1][0]:
        sp[0], sp[1] = sp[1], sp[0]

    if sp[2][0] > sp[3][0]:
        sp[2], sp[3] = sp[3], sp[2]

    return sp


def imgcorr(src):
    img = cv2.resize(src, (2000, 1500), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # image_show("gray", gray)

    edges = cv2.Canny(gray, 150, 100, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150,
                            minLineLength=200, maxLineGap=25)
    result1 = img.copy()
    # print(lines)
    # for i in range(int(np.size(lines) / 4)):
    print(int(np.size(lines) / 4))
    for i in range(int(np.size(lines) / 4)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(result1, (x1, y1), (x2, y2), (0, 0, 255), 2)
    image_show("", result1)

    points = np.zeros((4, 2), dtype="float32")
    points[0] = CrossPoint(lines[0], lines[2])
    points[1] = CrossPoint(lines[0], lines[3])
    points[2] = CrossPoint(lines[1], lines[2])
    points[3] = CrossPoint(lines[1], lines[3])
    print(points)

    cv2.circle(img, (points[0][0], points[0][1]), 2, (0, 255, 0), 10)
    cv2.circle(img, (points[1][0], points[1][1]), 2, (0, 255, 0), 10)
    cv2.circle(img, (points[2][0], points[2][1]), 2, (0, 255, 0), 10)
    cv2.circle(img, (points[3][0], points[3][1]), 2, (0, 255, 0), 10)

    image_show("points", img)

    sp = SortPoint(points)

    width = int(
        np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2))
    height = int(
        np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2))

    dstrect = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]], dtype="float32")

    transform = cv2.getPerspectiveTransform(np.array(sp), dstrect)
    warpedimg = cv2.warpPerspective(src, transform, (width, height))

    return warpedimg


if __name__ == '__main__':
    src = cv2.imread("pt.jpg")
    dst = imgcorr(src)
    # image_show("Image", dst)
    cv2.imwrite("output.jpg", dst)
