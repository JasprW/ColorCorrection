# python动态边缘检测
import cv2


def nothing(x):
    pass


img = cv2.imread('images/IMG_0793.jpg')
# img = cv2.imread('ceshi2.bmp')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# 范围可以根据需求，自己更改
cv2.createTrackbar('Val', 'image', 0, 500, nothing)
edges = cv2.Canny(img, 100, 200)

while(1):
    k = cv2.waitKey(1) & 0xFF
    # 当按esc键 触发关闭事件
    if k == 27:
        break

    Val = cv2.getTrackbarPos('Val', 'image')
    # 这里我直接将maxVal的值设置为minVal+100，可以根据需求自行更改
    edges = cv2.Canny(img, Val, Val + 100)

    cv2.imshow('image', edges)
cv2.destroyAllWindows()
