# @Author: Jaspr
# @Date:   2018-12-11T22:08:36+08:00
# @Email:  wang@jaspr.me
# @Last modified by:   Jaspr
# @Last modified time: 2018-12-12, 16:19:46

import MySQLdb
import sys
# import ConfigParser
import cv2
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')


def connectDB():
    '''
    连接数据库
    输出：【成功连接返回连接符，失败返回空数组】
    '''
    try:
        conn = MySQLdb.connect(
            host='localhost',
            user='root',
            passwd='Qy19941121!',
            db='ying',
            charset='utf8',
            connect_timeout=24 * 7200,
        )
    except Exception, e:
        print e
        return []

    conn.ping(True)
    cur = conn.cursor()

    return conn, cur


def getOriginalImage(imgID, dbImgName):
    '''
    获取数据库中图片
    输入：【imgID是表的主键，dbImgName是需要读取的图片的属性名】
    输出：【图像二进制文件，输出参数为-1表示读取原始图像失败】
    '''
    conn, cur = connectDB()
    # 获取imgID对应的图像二进制文件
    sqlGetImg = "SELECT %s FROM tb_tongueimage \
                        WHERE ID ='%d' LIMIT 1 " % (dbImgName, imgID)
    img = -1
    try:
        cur.execute(sqlGetImg)
        results = cur.fetchall()
        if len(results) > 0:
            if len(results[0]) > 0:
                numpyArray = np.fromstring(results[0][0], dtype=np.uint8)
                img = numpyArray.reshape(1440, 1080, 3)
    except Exception, e:
        print e
        print("get image from db error")
    # 关闭db
    cur.close()
    conn.close()

    return img


def insertImageToDB(imgID, dbImgName, imgFile):
    '''
    插入矫正后的图片
    输入：【imgID是表的主键，dbImgName是需要读取的图片的属性名,imgFile是需要插入的二进制文件】
    输出：【是否插入成功，输出参数0成功，输出参数-1插入失败】
    '''
    binaryImg = imgFile.tobytes()

    conn, cur = connectDB()
    conn.ping(True)
    sqlInsertImg = "UPDATE tb_tongueimage \
                    SET %s='%s'  WHERE ID='%d' LIMIT 1 " \
                    % (dbImgName, MySQLdb.escape_string(MySQLdb.Binary(binaryImg)), imgID)
    ret = 0
    try:
        cur.execute(sqlInsertImg)
        conn.commit()
    except Exception, e:
        print ("update question error")
        print e
        conn.rollback()
        ret = -1
    # 关闭db
    cur.close()
    conn.close()
    return ret


# 测试
if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    # 测试插入
    insertImageToDB(1, 'imageDataA', img)
    getOriginalImage(1, 'imageDataA')
