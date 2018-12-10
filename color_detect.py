import cv2
import numpy as np


def _img_split_with_shadow(gray_img, threshold_value=180):
    """
    :param binary_img: 读入的灰度图
    :param img_show:
    :return: 水平和垂直线的坐标集合
    """
    h = gray_img.shape[0]
    w = gray_img.shape[1]

    # 按行求和
    sum_x = np.sum(gray_img, axis=1)
    # 按列求和
    sum_y = np.sum(gray_img, axis=0)

    h_line_index = np.argwhere(sum_x == 0)
    v_line_index = np.argwhere(sum_y == 0)

    h_line_index = np.reshape(h_line_index, (h_line_index.shape[0],))
    v_line_index = np.reshape(v_line_index, (v_line_index.shape[0],))

    h_line = []
    v_line = []

    for i in range(len(h_line_index) - 1):
        if h_line_index[i + 1] - h_line_index[i] > 2:
            h_line.append((0, h_line_index[i + 1], w - 1, h_line_index[i + 1]))
            h_line.append((0, h_line_index[i], w - 1, h_line_index[i]))

    for i in range(len(v_line_index) - 1):
        if v_line_index[i + 1] - v_line_index[i] > 2:
            v_line.append((v_line_index[i + 1], 0, v_line_index[i + 1], h - 1))
            v_line.append((v_line_index[i], 0, v_line_index[i], h - 1))

    return h_line, v_line


def _combine_rect(h_lines, v_lines):
    """
    :param h_lines: 平行直线集合
    :param v_lines: 垂直直线集合
    :return: 返回由 h_lines 和 v_lines 组成的矩形集合
    """
    rects = []

    x_axis = sorted(set([item[0] for item in v_lines]))
    y_axis = sorted(set([item[1] for item in h_lines]))

    point_list = []
    for y in y_axis:
        point = []
        for x in x_axis:
            point.append((y, x))
        point_list.append(point)

    for y_index in range(len(y_axis) - 1):
        for x_index in range(len(x_axis) - 1):
            area = abs((y_axis[y_index + 1] - y_axis[y_index]) * (x_axis[x_index + 1] - x_axis[x_index]))
            rects.append([(y_axis[y_index], x_axis[x_index],
                           y_axis[y_index + 1], x_axis[x_index + 1]), area])
    # 按面积降序排序
    rects.sort(key=lambda ele: ele[1], reverse=True)
    areas = [ele[1] for ele in rects]

    # 找到相邻差值最大的序号
    max = -1
    index = 0
    for i in range(len(areas) - 1):
        dif = areas[i] - areas[i + 1]
        if max < dif:
            max = dif
            index = i + 1

    # rects 按坐标升序排序，使得颜色顺序和标准色卡一致
    rect_list = [ele[0] for ele in rects[0:index]]
    rect_list.sort(key=lambda ele: ele[1])
    rect_list.sort(key=lambda ele: ele[0])

    # for i in range(len(rect_list) - 1):
    #     for j in range(0, len(rect_list) - 1 - i):
    #         if rect_list[j + 1][1] < rect_list[j][1] :
    #             rect_list[j], rect_list[j + 1] = rect_list[j + 1], rect_list[j]
    #
    # for i in range(len(rect_list) - 1):
    #     for j in range(0, len(rect_list) - 1 - i):
    #         if rect_list[j + 1][0] < rect_list[j][0]:
    #             rect_list[j], rect_list[j + 1] = rect_list[j + 1], rect_list[j]

    return rect_list


def image_show(name, img):
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
            data = img[i - 2:i + 2, j - 2:j + 2]
            b, g, r = cv2.split(data)
            color_matrix.append([int(np.mean(b)), int(np.mean(g)), int(np.mean(r))])
    color_matrix = np.array(color_matrix)
    return color_matrix


def img_split(img, img_show=False):
    """
    分割待测试的色卡图像，返回分割后的矩形图像列表和回归方程所需要的输入图像 shape:(4,6,3),像素格式：(b,g,r)
    :param img_file: 待测试色卡图像
    :param img_show: 是否显示
    :return: 分割后的子图像rect列表
    """
    # 四周各填充10个像素
    padding = 10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    binary = cv2.blur(binary, (5, 5))
    # 反色
    binary = cv2.bitwise_not(binary)
    # cv2.imshow('cece', binary)
    # cv2.waitKey()
    binary = cv2.copyMakeBorder(
        binary, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    h = img.shape[0]
    w = img.shape[1]
    rate = h // w if h > w else w // h

    h_line_shadow, v_line_shadow = _img_split_with_shadow(binary)
    h_line = h_line_shadow
    v_line = v_line_shadow
    # print("h_line:\n", h_line, "\nv_line:\n", v_line)
    rects = _combine_rect(h_line, v_line)
    # print(rects)
    split_imgs = []

    # padding过，所以定位的时候需要减去padding的值
    img = cv2.copyMakeBorder(
        img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    color_img = np.zeros((4, 6, 3), dtype=np.uint8)
    for index, rect in enumerate(rects):
        rect_img = img[rect[0]:rect[2], rect[1]:rect[3]]
        color_img[index // 6][index % 6] = get_center_color(rect_img)
        # print(index, color_img[index//6][index%6])
        split_imgs.append(rect_img)

    if img_show:
        p = 1
        for rect in rects:
            cv2.rectangle(img, (rect[1], rect[0]),
                          (rect[3], rect[2]), (0, 255, 0), 2)
            # 给识别对象写上标号
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 加减10是调整字符位置
            cv2.putText(
                img, str(p), (rect[1] - 10, rect[0] + 10), font, 1, (0, 0, 255), 2)
            p += 1

        img = cv2.resize(img, (int(h * 0.7), int(h * 0.7 / rate)))
        # cv2.imshow('cece', img)
        # cv2.waitKey()
        image_show('img', img)

    # image_show('color_img', color_img)
    return split_imgs, color_img


def get_center_color(img):
    """
    计算给定图像中间（5，5）像素的均值
    :param img:
    :return:
    """
    w = img.shape[0]
    w = w // 2
    h = img.shape[1]
    h = h // 2
    data = img[h - 2:h + 2, w - 2:w + 2]
    b, g, r = cv2.split(data)
    return (int(np.mean(b)), int(np.mean(g)), int(np.mean(r)))
