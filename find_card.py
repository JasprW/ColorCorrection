# @Author: Jaspr
# @Date:   2018-11-29T13:19:21+08:00
# @Email:  wang@jaspr.me
# @Last modified by:   Jaspr
# @Last modified time: 2018-12-17, 20:39:49

import cv2
import numpy as np
import os
import sys


def image_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_point(points):
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


def is_duplicate(c, contours):
    """
    判断当前轮廓是否与其他轮廓属于同一定位点，用于筛选同一定位点的唯一轮廓
    :param c: 当前判断的contour
    :param contours: 当前所有非重叠contours集合
    :return: contours中是否存在与c相交的轮廓，bool，重复轮廓返回True，不重复返回False
    """
    if contours == []:
        return False
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


def find_corner(img):
    """
    获取色卡四角的定位点
    :param img: 输入图像
    :return: 四个角点坐标
    """
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(
    #     gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # 使用自适应二值化避免过曝影响定位点识别
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 使用膨胀让边界清晰，避免轮廓断裂
    # TODO: 尝试形态学开/闭运算解决边缘断开问题
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = binary
    dilated = cv2.dilate(dilated, kernel)

    blur = cv2.GaussianBlur(dilated, (9, 9), 0)
    edges = cv2.Canny(blur, 100, 300)

    _, contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return []

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
            if (rate >= 0.7 and cv2.contourArea(contours[i]) / (w * h) >= 0.8):
                # cv2.drawContours(img_test, contours, i,
                #                  (255, 255, 0), 1)

                # 判断轮廓层级，筛选多层轮廓的外围轮廓
                while hierarchy[k][2] != -1:
                    k = hierarchy[k][2]
                    c = c + 1

                # 超过n层则判断为定位点，默认4层
                if c == 3:
                    cv2.drawContours(img_test, contours, i,
                                     (255, 0, 0), 3, cv2.LINE_AA)
                if c >= 4:
                    cv2.drawContours(img_test, contours, i,
                                     (0, 0, 255), 3, cv2.LINE_AA)
                    if i not in found:
                        found.append(i)
                    continue
        else:
            continue

    temp_contours = []
    for i in found:
        temp_contours.append(contours[i])

    img_dc = img.copy()

    # 按轮廓面积从大到小排序
    contours = sorted(
        temp_contours, key=cv2.contourArea, reverse=True)

    if len(contours) < 4:
        return []

    candidate_contours = []

    for contour in contours:
        if is_duplicate(contour, candidate_contours):
            continue
        else:
            candidate_contours.append(contour)
            if len(candidate_contours) >= 4:
                break

    # print(len(candidate_contours))
    if len(candidate_contours) < 4:
        # print("仅找到", len(candidate_contours), "个定位点")
        # image_show("binary", binary)
        # image_show("", edges)
        # image_show("test", img_test)
        # for i in range(len(contours)):
        #     cv2.drawContours(img_dc, contours, i,
        #                      (255, 255, 0), 5, cv2.LINE_AA)
        # image_show("img_dc", img_dc)
        return []

    location_points = []

    for i in range(4):
        pos_rect = cv2.minAreaRect(candidate_contours[i])
        location_points.append(pos_rect[0])

    # 对定位点排序，排序后顺序为：左上，右上，左下，右下
    location_points = sort_point(location_points)
    return location_points


def get_color_card(img, points):
    """
    通过角点提取色卡部分，并对色卡进行透视校正，返回校正后色卡正视图
    :param img: 输入图像
    :param points: 色卡角点坐标
    :return: 透视校正后的色卡图片
    """
    # 对定位点排序，排序后顺序为：左上，右上，左下，右下
    sp = sort_point(points)

    # 查看定位点位置及顺序
    # img_point_position = img.copy()
    # for i in range(0, 4):
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.circle(img_point_position, (np.float32(sp[i][0]), np.float32(
    #         sp[i][1])), 20, (255, 255, 0), -1, cv2.LINE_AA)
    #     cv2.putText(img_point_position, str(
    #         i + 1), (np.int0(sp[i][0]) - 10, np.int0(sp[i][1]) + 10), font, 10, (255, 255, 0), 10)
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
        path = sys.argv[1]
        # 传入文件夹路径
        if os.path.isdir(path):
            dir_path = path
            file_num = 32
            success_num = 0
            fail_num = 0
            for i in range(file_num + 1):
                file_path = dir_path + '/' + str(i) + '.jpg'
                if os.path.isfile(file_path):
                    img = cv2.imread(file_path)
                    corner_points = find_corner(img)

                    if corner_points == []:
                        fail_num += 1
                        print('[' + str(i) + ']', '定位失败，未找到足够定位点！')
                        # 将未识别到色卡的照片统一存储至fail文件夹
                        detect_fail_dir = dir_path + '/fail-test-2'
                        if not os.path.isdir(detect_fail_dir):
                            os.makedirs(detect_fail_dir)
                        cv2.imwrite(detect_fail_dir + '/' + str(i) + '.jpg', img)
                        continue

                    card = get_color_card(img, corner_points)
                    cv2.imwrite(dir_path + '/' + str(i) + '-card-test-2' + '.jpg', card)
                    print('[' + str(i) + ']', '找到色卡！')
                    success_num += 1
            print('success:', success_num)
            print('fail:', fail_num)
            print('rate:', success_num / file_num)

        # 传入文件路径
        elif os.path.isfile(path):
            file_path = path
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_ext = os.path.splitext(os.path.basename(file_path))[1]
            dir_name = os.path.dirname(file_path)
            img = cv2.imread(file_path)
            corner_points = find_corner(img)

            if corner_points == []:
                print("未找到定位点！")
                # 将未识别到色卡的照片统一存储至fail文件夹
                detect_fail_dir = dir_name + '/fail-test'
                if not os.path.isdir(detect_fail_dir):
                    os.makedirs(detect_fail_dir)
                cv2.imwrite(detect_fail_dir + '/' + file_name + file_ext, img)
                sys.exit()

            card = get_color_card(img, corner_points)
            cv2.imwrite(dir_name + '/' + file_name + '-card-test' + file_ext, card)
            print("找到色卡！")

        else:
            print("参数错误！未找到对应文件或文件夹")
            sys.exit()
    else:
        print("参数数量错误！")
    # image_show('', card)
