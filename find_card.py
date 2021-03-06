# @Author: Jaspr
# @Date:   2018-11-29T13:19:21+08:00
# @Email:  wang@jaspr.me
# @Last modified by:   Jaspr
# @Last modified time: 2019-03-15, 14:26:59

import os
import sys
import platform
import cv2
import numpy as np
from card_verify import is_card_ok


def image_show(name, img):
    """
    GUI显示图片
    :param name: 窗口标题
    :img: 需要显示的图片
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_color(color_card):
    """
    获取实际拍摄的照片中色卡各颜色色值
    :param color_card: 透视校正完成的色卡图片
    :return: 色卡中的颜色色值，以矩阵格式存储，color_matrix，shape: (3 , 24)
    """
    img = color_card.copy()
    color_matrix = []
    pos_hori = [34, 100, 166, 232, 298, 364]
    pos_vert = [44, 116, 190, 260]
    for i in pos_vert:
        for j in pos_hori:
            data = img[i - 3:i + 3, j - 3:j + 3]
            b, g, r = cv2.split(data)
            color_matrix.append([int(np.mean(b)), int(np.mean(g)), int(np.mean(r))])
    color_matrix = np.array(color_matrix)
    return color_matrix


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


def is_rect(contour, rate=0.8):
    """
    判断轮廓是否为矩形
    :param contour: 对其判断的轮廓
    :param rate: 轮廓长短边比值的判断阈值
    :return: 轮廓contour是否为矩形，bool，是矩形返回True，不是返回False
    """
    rect = cv2.minAreaRect(contour)
    w = rect[1][0]
    h = rect[1][1]
    if w and h:
        rate = min(w, h) / max(w, h)
    if cv2.contourArea(contour) / w * h < 0.8 or rate < 0.75:
        return False
    else:
        return True


def _intersection(a, b):
    """
    判断两个rect是否相交
    :param a: rect1
    :param b: rect2
    :return: 是否相交，bool，相交为True，不相交为False
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False
    return True


def find_corner(img, blockSize=13, param_c=2, debug=False):
    """
    获取色卡四角的定位点
    :param img: 输入图像
    :param blockSize: Adaptive Threshold 参数 blockSize
    :param param_c: Adaptive Threshold 参数 C
    :param debug: 是否使用debug模式，输出各步骤结果和图片
    :return: 四个角点坐标
    """
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    # 使用自适应二值化避免过曝影响定位点识别
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize, param_c)

    # 形态学闭运算解决边缘断开问题
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    blur = binary
    edges = cv2.Canny(blur, 100, 300)

    _, contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    found = []
    found = set(found)

    img_test = img.copy()

    for i in range(len(contours)):
        k = i
        c = 0

        # 获取轮廓包围框及其长宽
        rect = cv2.minAreaRect(contours[i])
        w = rect[1][0]
        h = rect[1][1]
        # 选取方形轮廓
        if w and h and (is_rect(contours[i])):
            # 判断轮廓层级，筛选多层轮廓的外围轮廓
            while hierarchy[k][2] != -1:
                k = hierarchy[k][2]
                c = c + 1

                # 超过n层则判断为定位点，默认4层
                if c >= 4:
                    found.add(i)
                    break

    temp_contours = []
    for i in found:
        temp_contours.append(contours[i])

    # 按轮廓面积从大到小排序
    contours = sorted(
        temp_contours, key=cv2.contourArea, reverse=True)

    if len(contours) < 4:
        if debug is True:
            for i in range(len(contours)):
                cv2.drawContours(img_test, contours, i,
                                 (0, 0, 255), 4, cv2.LINE_AA)
            image_show("test", img_test)
        return []

    candidate_contours = []

    for c in contours:
        if is_duplicate(c, candidate_contours):
            continue
        else:
            candidate_contours.append(c)
            if len(candidate_contours) >= 5:
                break

    # 根据轮廓面积排除色块轮廓
    if cv2.contourArea(candidate_contours[0]) / cv2.contourArea(candidate_contours[1]) > 1.4:
        candidate_contours.pop(0)
    else:
        candidate_contours = candidate_contours[0:4]

    if debug is True:
        for i in range(len(contours)):
            cv2.drawContours(img_test, contours, i,
                             (0, 0, 255), 4, cv2.LINE_AA)
        image_show("test", img_test)

    if len(candidate_contours) < 4:
        if debug:
            print("仅找到", len(candidate_contours), "个定位点")
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

    # 设定裁剪边距，完全去除定位标志
    padding = np.int0(warpedimg.shape[0] * 0.06)
    img_cropped = warpedimg[padding:(
        warpedimg.shape[0] - padding), padding:(warpedimg.shape[1] - padding)]
    img_output = cv2.resize(img_cropped, (400, 300),
                            interpolation=cv2.INTER_CUBIC)

    return img_output


def retry(img, debug=False):
    """
    找不到色卡或色卡不正确时调整 Adaptive Threshold 的参数进行重试
    :param img: 待寻找色卡的图片
    :param debug: 是否使用debug模式，输出中间步骤结果
    :return: 未找到足够定位点返回-1，找到的色卡不正常返回-2，正常返回色卡图片（矩阵）
    """
    blockSize = 19
    param_c = 1
    retry_time = 6
    corner_points = []
    card = 0
    for i in range(retry_time):
        if debug:
            print('第', i + 1, '次重试...')
        blockSize = blockSize - 2
        param_c = 2 if blockSize < 10 else 1
        corner_points = find_corner(img, blockSize, param_c, debug)
        if corner_points == []:
            if not isinstance(card, int):
                break
            else:
                continue
        card = get_color_card(img, corner_points)
        if not is_card_ok(card):
            if debug:
                print('找到色卡但错误,重试...')
            continue
        else:
            return card

    if corner_points == [] and isinstance(card, int):
        return -1
    elif not isinstance(card, int) and is_card_ok(card):
        return card
    else:
        if debug:
            image_show('', card)
            cv2.imwrite(card_dir + slash + file_name + '-wrong.jpg', card)
        return -2


if __name__ == '__main__':
    slash = '\\' if platform.system() == "Windows" else '/'

    if len(sys.argv) != 2:
        print("参数数量错误！")
        sys.exit()

    path = sys.argv[1]
    if not os.path.isdir(path) and not os.path.isfile(path):
        print("参数错误！未找到对应文件或文件夹")
        sys.exit()

    # 传入文件夹路径
    elif os.path.isdir(path):
        dir_path = path
        card_dir = dir_path + slash + 'card'
        wrong_card_dir = card_dir + slash + 'wrong'
        fail_dir = dir_path + slash + 'fail'
        files = os.listdir(dir_path)
        img_files = []
        for f in files:
            if not os.path.isdir(f):
                file_name, file_ext = os.path.splitext(os.path.basename(f))
                if file_ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                    img_files.append(f)
        success_num = 0
        fail_num = 0
        wrong_num = 0
        for i in range(len(img_files)):
            file_name, file_ext = os.path.splitext(img_files[i])
            file_path = dir_path + slash + img_files[i]
            img = cv2.imread(file_path)
            print('[' + img_files[i] + ']')
            retry_result = 0
            corner_points = find_corner(img)

            if corner_points == []:
                # 失败重试...
                retry_result = retry(img)
                if isinstance(retry_result, int) and retry_result == -1:
                    fail_num += 1
                    print('定位失败，未找到足够定位点！')
                    # 将未识别到色卡的照片统一存储至fail文件夹
                    if not os.path.isdir(fail_dir):
                        os.makedirs(fail_dir)
                    cv2.imwrite(fail_dir + slash + file_name + '-fail.jpg', img)
                    continue

            card = get_color_card(img, corner_points) if (isinstance(retry_result, int) and retry_result == 0) else retry_result

            # 检查卡片是否正常
            if not is_card_ok(card):
                if not retry_result:
                    card = retry(img)
                if not is_card_ok(card):
                    print('找到色卡，但色卡提取不正确！')
                    if not os.path.isdir(fail_dir):
                        os.makedirs(fail_dir)
                    if not os.path.isdir(wrong_card_dir):
                        os.makedirs(wrong_card_dir)
                    cv2.imwrite(fail_dir + slash + file_name + '-wrong.jpg', img)
                    cv2.imwrite(wrong_card_dir + slash + file_name + '-card.jpg', card)
                    wrong_num += 1
                    continue
            print('找到色卡！')
            if not os.path.isdir(card_dir):
                os.makedirs(card_dir)
            cv2.imwrite(card_dir + slash + file_name + '-card.jpg', card)
            success_num += 1

        print('success:', success_num)
        print('fail:', fail_num)
        print('wrong:', wrong_num)
        print('rate:', success_num / len(img_files))

    # 传入文件路径
    elif os.path.isfile(path):
        file_path = path
        file_name, file_ext = os.path.splitext(os.path.basename(file_path))
        dir_path = os.path.dirname(file_path)
        card_dir = dir_path + slash + 'card'
        fail_dir = dir_path + slash + 'fail'

        img = cv2.imread(file_path)

        corner_points = find_corner(img, debug=True)

        retry_result = 0
        if not corner_points:
            retry_result = retry(img, debug=True)
            if isinstance(retry_result, int):
                if not os.path.isdir(fail_dir):
                    os.makedirs(fail_dir)
                if retry_result == -1:
                    print("未找到定位点！")
                    # 将未识别到色卡的照片统一存储至fail文件夹
                    cv2.imwrite(fail_dir + slash + file_name + '-fail.jpg', img)
                else:
                    cv2.imwrite(fail_dir + slash + file_name + '-wrong.jpg', img)
                    print("不正常")
                sys.exit()

        card = get_color_card(img, corner_points) if (isinstance(retry_result, int) and retry_result == 0) else retry_result
        # 检查卡片是否正常
        if not is_card_ok(card):
            if not retry_result:
                card = retry(img, debug=True)
            if isinstance(card, int) or not is_card_ok(card):
                cv2.imwrite(fail_dir + slash + file_name + '-wrong.jpg', img)
                print("不正常")
                sys.exit()
        if not os.path.isdir(card_dir):
            os.makedirs(card_dir)
        cv2.imwrite(card_dir + slash + file_name + '-card' + '.jpg', card)
        print("找到色卡！")
