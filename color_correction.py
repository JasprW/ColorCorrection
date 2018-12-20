# @Author: Jaspr
# @Date:   2018-11-29T10:11:16+08:00
# @Email:  wang@jaspr.me
# @Last modified by:   Jaspr
# @Last modified time: 2018-12-20, 14:49:24

from find_card import *
from card_direction_detect import *
import numpy as np
import cv2
import sys
import os

std_color_file = 'color_value_test.csv'


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


def get_A_matrix(x, y):
    """
    生成系数矩阵A
    :param x: 输入数据,shape:(16, n)
    :param y: 样本的标准数据, shape:(3,n)
    :return: 返回训练好的系数矩阵A， shape: (3 , 16)
    """
    temp1 = np.dot(x, x.T)
    temp2 = np.dot(x, y.T)
    temp1 = np.linalg.inv(temp1)
    A = np.dot(temp1, temp2)
    return A.T


def get_polynomial(R, G, B):
    """
    根据RGB值生成多项式
    :param rgb: 像素点的RGB值,格式(r,g,b)
    :return: 返回构造的多项式
    """
    R = int(R)
    G = int(G)
    B = int(B)

    # 十项多项式（灰度偏淡）
    # return [1, R, G, B, R * G, R * B, B * G, R * R, B * B, G * G]

    # 九项多项式（灰度偏深）
    return [R, G, B, R * G, R * B, B * G, R * R, B * B, G * G]

    # 十五项多项式
    # return [R, G, B,
    #         R * G, R * B, B * G, R * R, B * B, G * G,
    #         R * (G * G), R * (B * B), G * (R * R), G * (B * B), B * (R * R), B * (G * G)]

    # 十六项多项式
    # return [R, G, B,
    #     R * G, R * B, B * G, R * R, B * B, G * G,
    #     R * (G * G), R * (B * B), G * (R * R), G * (B * B), B * (R * R), B * (G * G),
    #     R * G * B]


def img_digitization(image_data):
    """
    将图片电子化，生成色值矩阵
    :param image_data: 待校正的原始图片
    :return: 返回线性回归需要的输入矩阵, shape:(n, image_data.shape[0] * image_data.shape[1]), n为多项式项数
    """
    # FIXME: 解决内存占满问题，可能存在内存泄漏
    # TODO: 尝试每行/列读取校正
    data = []
    for raw_data in image_data:
        for bgr in raw_data:
            data.append(get_polynomial(bgr[2], bgr[1], bgr[0]))
    data = np.array(data)
    print(sys.getsizeof(image_data))
    print(sys.getsizeof(data))
    return data.T


def create_inputData(color_data):
    """

    :param color_data: 待校正的色卡颜色数据，shape:(24, 3)
    :return: 返回线性回归需要的输入矩阵, shape:(n, 24), n为多项式项数
    """
    data = []
    for bgr in color_data:
        data.append(get_polynomial(bgr[2], bgr[1], bgr[0]))

    data = np.array(data)
    return data.T


def get_stdColor_value():
    """
    构造标准色卡的R,G,B矩阵，shape: (3 , 24)
    :return: 返回标准色卡的色值，用矩阵存储[B, G, R]
    """
    color_dict = {}
    std_matrix = []
    color_value_list = np.loadtxt(std_color_file, dtype=np.str, delimiter=',')

    for element in color_value_list:
        color_dict[element[1]] = (
            int(element[2]), int(element[3]), int(element[4]))
        std_matrix.append([int(element[2]), int(element[3]), int(element[4])])

    std_matrix = np.array(std_matrix)
    # print(std_matrix.shape)
    # print("std_matrix:\n", std_matrix)
    return std_matrix.T


def recorrect_color(raw_img, A):
    """
    用系数矩阵A对图像进行颜色校正
    :param raw_img: 原始图像
    :param A: 系数矩阵
    :return: 返回校正后的图像
    """
    w = raw_img.shape[0]
    h = raw_img.shape[1]
    input_data = img_digitization(raw_img)
    corrected_data = np.dot(A, input_data)
    data = []
    for element in corrected_data:
        vec = []
        for value in element:
            if 0.0 <= value <= 255.0:
                vec.append(int(value))
            elif 0.0 > value:
                vec.append(0)
            elif 255.0 < value:
                vec.append(255)
        data.append(vec)

    data = np.array(data)
    data = data.transpose((1, 0))
    corrected_img = data.reshape((w, h, 3))

    return corrected_img


if __name__ == '__main__':
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_ext = os.path.splitext(os.path.basename(file_path))[1]
            dir_name = os.path.dirname(file_path)
        else:
            print("未找到文件")
    else:
        print("参数数量错误")

    # 载入标准色卡数据
    std_matrix = get_stdColor_value()
    # real_matrix = get_realColor_value()

    # 载入测试色卡图像，生成回归输入数据
    img = cv2.imread(file_path)

    # 定位色卡并进行透视变换为正视图
    points = find_corner(img)
    if not points:
        # 替换参数重试一次
        points = find_corner(img, b=1)
        if not points:
            print("未找到定位点！")
    else:
        color_card = get_color_card(img, points)

        # 检测色卡是否翻转、镜像或翻转+镜像，并对其进行相应变换
        if is_upsideDown_and_mirrorred(color_card):
            color_card = rotate(color_card)
            color_card = cv2.flip(color_card, 1)
        if is_upsideDown(color_card):
            color_card = rotate(color_card)
        if is_mirrored(color_card):
            color_card = cv2.flip(color_card, 1)
        # image_show("card", color_card)

        # 使用extract_color获取各色块中心颜色
        color_data = extract_color(color_card)
        input_data = create_inputData(color_data)

        A = get_A_matrix(input_data, std_matrix)

        # 颜色校正
        img_resized = cv2.resize(img.copy(), None, fx=0.5, fy=0.5)
        corrected_img = recorrect_color(img, A)
        # cv2.imwrite('output/corrected.jpg', corrected_img[..., [2, 1, 0]])

        output_dir = dir_name + '/output'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        img = cv2.imread(file_path)
        cv2.imwrite(output_dir + '/' + file_name + '-corrected' + file_ext, corrected_img[..., [2, 1, 0]])

        print("Color correction complete!")
