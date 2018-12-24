# @Author: Jaspr
# @Date:   2018-12-10, 15:29:33
# @Email:  wang@jaspr.me
# @Last modified by:   Jaspr
# @Last modified time: 2018-12-18, 15:04:05

from find_card import *
from color_correction import extract_color
import cv2


def is_upsideDown(color_card):
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
    real_color = extract_color(color_card)

    upper_left = real_color[0]
    upper_right = real_color[5]
    bottom_left = real_color[18]
    bottom_right = real_color[23]

    return bool(bottom_left[0] < upper_left[0] < bottom_right[0] and bottom_left[2] < upper_right[2] < bottom_right[2])


def is_upsideDown_and_mirrorred(color_card):
    real_color = extract_color(color_card)

    upper_left = real_color[0]
    upper_right = real_color[5]
    bottom_left = real_color[18]
    bottom_right = real_color[23]

    if upper_left[0] > bottom_left[0] > upper_right[0] and upper_left[2] > bottom_right[2] > upper_right[2]:
        return True
    else:
        return False


def rotate(img, angle=180, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


if __name__ == '__main__':
    img = cv2.imread('images/Image2.jpeg', 1)
    sp = find_corner(img)
    card = get_color_card(img, sp)

    rotated = rotate(card)
    mirrored = cv2.flip(card, 1)
    rotated_mirrorred = cv2.flip(rotated, 1)

    print(is_upsideDown(rotated))
    print(is_mirrored(rotated))
    print(is_upsideDown_and_mirrorred(rotated), '\n')

    print(is_upsideDown(mirrored))
    print(is_mirrored(mirrored))
    print(is_upsideDown_and_mirrorred(mirrored), '\n')

    print(is_upsideDown(rotated_mirrorred))
    print(is_mirrored(rotated_mirrorred))
    print(is_upsideDown_and_mirrorred(rotated_mirrorred))
