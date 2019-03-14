# @Author: Jaspr
# @Date:   2018-12-10, 15:29:33
# @Email:  wang@jaspr.me
# @Last modified by:   Jaspr
# @Last modified time: 2019-03-14, 15:15:29

import sys
import os
import cv2


def is_upsideDown(color_card):
    from find_card import extract_color
    real_color = extract_color(color_card)

    upper_left = real_color[0]
    upper_right = real_color[5]
    bottom_left = real_color[18]
    bottom_right = real_color[23]
    # print(upper_left, upper_right, bottom_left, bottom_right)
    # print(real_color[0], real_color[1], real_color[2], real_color[3], real_color[4], real_color[5])

    if (upper_right >= bottom_left).all() and (upper_right >= bottom_right).all() and (upper_right >= upper_left).all():
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

    if (bottom_right >= bottom_left).all() and (bottom_right >= upper_left).all() and (bottom_right >= upper_right).all():
        return True
    else:
        return False


def is_upsideDown_and_mirrorred(color_card):
    from find_card import extract_color
    real_color = extract_color(color_card)

    upper_left = real_color[0]
    upper_right = real_color[5]
    bottom_left = real_color[18]
    bottom_right = real_color[23]

    if (upper_left >= bottom_left).all() and (upper_left >= bottom_right).all() and (upper_left >= upper_right).all():
        return True
    else:
        return False


def is_card_ok(color_card, debug=False):
    # from find_card import image_show
    gray = cv2.cvtColor(color_card.copy(), cv2.COLOR_BGR2GRAY)

    # 使用自适应二值化避免过曝影响定位点识别
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 1)
    blur = binary
    edges = cv2.Canny(blur, 50, 150)

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
        w = rect[2]
        h = rect[3]
        if cv2.contourArea(contours[i]) / (w * h) < 0.87:
            c += 1
    if c >= 4:
        if debug:
            from find_card import image_show
            image_show('', img_test)
            print(c)
        return False
    else:
        if debug:
            from find_card import image_show
            image_show('', img_test)
            print(c)
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
        dir_path = path
        # 传入文件夹路径
        if os.path.isdir(path):
            dir_path = path
            img_files = []
            files = os.listdir(dir_path)
            for f in files:
                if not os.path.isdir(f):
                    file_name, file_ext = os.path.splitext(os.path.basename(f))
                    if file_ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                        img_files.append(f)
            success_num = 0
            fail_num = 0

            for f in img_files:
                card = cv2.imread(dir_path + '/' + f)
                card_rotated = rotate(card)
                print(f, is_card_ok(card))
                # if not is_card_ok(card):
                #     print("卡片不正常！")
                #     sys.exit()
                # else:
                #     print("卡片正常！")
                # if not is_upsideDown(card) and is_upsideDown(card_rotated):
                #     print("正常！")
                #     success_num += 1
                #     sys.exit()
                # else:
                #     print("失败！", f)
                #     fail_num += 1
            # print(success_num)
            # print(fail_num)

        if os.path.isfile(path):
            if(is_card_ok(cv2.imread(path), debug=True)):
                print('OK!')
                card = cv2.imread(path)
                # from find_card import extract_color
                # real_color = extract_color(card)
                # upper_left = real_color[0]
                # upper_right = real_color[5]
                # bottom_left = real_color[18]
                # bottom_right = real_color[23]
                # print(upper_left, upper_right, bottom_left, bottom_right)
                # print(is_upsideDown(card))
                # from find_card import image_show
                # image_show('', card)
            else:
                print('WRONG!!!')
