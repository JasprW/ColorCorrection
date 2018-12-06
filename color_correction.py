from color_detect import *
from find_card import *
import numpy as np
import cv2
import sys
import os

std_color_file = 'color_value.csv'
real_color_file = 'real_value.csv'


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
    # return [R, G, B, R*G, R*B, B*G, R*R, B*B, G*G]

    # 十五项多项式
    return [R, G, B,
            R * G, R * B, B * G, R * R, B * B, G * G,
            R * (G * G), R * (B * B), G * (R * R), G * (B * B), B * (R * R), B * (G * G)]

    # 十六项多项式
    # return [R, G, B,
    #     R * G, R * B, B * G, R * R, B * B, G * G,
    #     R * (G * G), R * (B * B), G * (R * R), G * (B * B), B * (R * R), B * (G * G),
    #     R * G * B]


def img_digitization(image_data):
    """

    :param image_data: 待校正的原始图片
    :return: 返回线性回归需要的输入矩阵, shape:(16, image_data.shape[0] * image_data.shape[1])
    """
    data = []
    for raw_data in image_data:
        for bgr in raw_data:
            data.append(get_polynomial(bgr[2], bgr[1], bgr[0]))
    data = np.array(data)
    return data.T


def create_inputData(color_data):
    """

    :param color_data: 待校正的色卡颜色数据，shape:(24, 3)
    :return: 返回线性回归需要的输入矩阵, shape:(15, 24)
    """
    data = []
    for bgr in color_data:
        data.append(get_polynomial(bgr[2], bgr[1], bgr[0]))

    data = np.array(data)
    return data.T


def get_stdColor_value():
    """
    构造标准色卡的R,G,B矩阵，shape: (3 , 24)
    :return: 返回标准色卡的R,G,B值，用矩阵存储
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


def get_realColor_value():
    """
    [TEST ONLY]构造实际拍摄的照片中色卡的R,G,B矩阵，shape: (3 , 24)
    :return: 返回实际颜色的R,G,B值，用矩阵存储
    """
    color_dict = {}
    std_matrix = []
    color_value_list = np.loadtxt(real_color_file, dtype=np.str, delimiter=',')

    for element in color_value_list:
        color_dict[element[1]] = (
            int(element[2]), int(element[3]), int(element[4]))
        std_matrix.append([int(element[2]), int(element[3]), int(element[4])])

    std_matrix = np.array(std_matrix)
    return std_matrix


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
    # print ('是否为文件:', path.isfile(sys.argv[1]))
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

    # 载入标准色卡数据
    std_matrix = get_stdColor_value()
    real_matrix = get_realColor_value()

    # 载入测试色卡图像，生成回归输入数据
    # img = cv2.imread('images/biased.jpg', 1)
    # img = cv2.imread('images/Image5.jpg', 1)
    img = cv2.imread(file_path, 1)

    # original
    # _, color_img = img_split(img)
    # input_data = create_inputData(color_img)
    # print("input_data:\n", input_data.shape)

    # 定位色卡并进行透视变换为正视图
    points = find_corner(img)
    color_card = get_color_card(img, points)
    image_show("card", color_card)

    # 使用extract_color()获取各色块中心颜色
    color_data = extract_color(color_card)
    input_data = create_inputData(color_data)

    A = get_A_matrix(input_data, std_matrix)

    # 颜色校正
    img_resized = cv2.resize(img.copy(), None, fx=0.5, fy=0.5)
    corrected_img = recorrect_color(img, A)
    # cv2.imwrite('output/corrected.jpg', corrected_img[..., [2, 1, 0]])

    output_dir = dir_name + '/output'
    if not os.path.isdir(output_dir):
        os.makedirs(dir_name + '/output')
    img = cv2.imread(file_path)
    cv2.imwrite(output_dir + '/' + file_name + '-corrected' + file_ext, corrected_img[..., [2, 1, 0]])

    print("Color correction complete!")