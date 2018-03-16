# Object Boundary Based Denoising for Depth Images
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def my_canny(gray, hole_index):
    projectPath = '/home/jianquan/PycharmProjects/ELG5163-Depth-Image-Denoising/'
    rawDataPath = projectPath + r'kinect-like/'
    colorimg = cv2.imread(rawDataPath + 'colorart.bmp')

    sigma1 = sigma2 = 0.1
    sum = 0

    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1)
                                                + (np.square(j - 3) / np.square(sigma2)))) / (
                                         2 * math.pi * sigma1 * sigma2)
            sum = sum + gaussian[i, j]

    gaussian = gaussian / sum

    # step1.高斯滤波

    W, H = gray.shape
    new_gray = np.zeros([W - 5, H - 5])
    for i in range(W - 5):
        for j in range(H - 5):
            new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

    # step2.增强 通过求梯度幅值
    W1, H1 = new_gray.shape
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值

    # plt.imshow(d, cmap="gray")

    # setp3.非极大值抑制 NMS
    W2, H2 = d.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):

            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]

                # 如果Y方向幅度值较大
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果X方向幅度值较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0

    # plt.imshow(NMS, cmap = "gray")

    # step4. 双阈值算法检测、连接边缘
    W3, H3 = NMS.shape
    print(W3,H3)
    DT = np.zeros([W3, H3])
    # 定义高低阈值
    TL = 0.2 * np.max(NMS)
    TH = 0.3 * np.max(NMS)
    # TL = 1
    # TH = 10

    for i in range(0, W3):
        for j in range(1, H3):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
                  or (NMS[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
            # if [i, j] not in hole_index:
            #     if (NMS[i, j] < TL):
            #         DT[i, j] = 0
            #     elif (NMS[i, j] > TH):
            #         DT[i, j] = 1
            #     elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any() or (NMS[i, [j - 1, j + 1]] < TH).any()):
            #         DT[i, j] = 1
            # else:
            #     TL = np.var([colorimg[i][j][0], colorimg[i][j][1], colorimg[i][j][2]]) * 0.00001
            #     if (NMS[i, j] < TL):
            #         DT[i, j] = 0
            #     elif (NMS[i, j] > TH):
            #         DT[i, j] = 1
            #     elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any() or (NMS[i, [j - 1, j + 1]] < TH).any()):
            #         DT[i, j] = 1

    plt.imshow(DT, cmap="gray")
    return DT


def Color_edge_denerator(img):

    edge = cv2.Canny(img, 60, 70)
    return edge

def Depth_edge_generator(depth):
    medianFilted = cv2.medianBlur(depth, 5)
    edge = cv2.Canny(medianFilted, 150, 200)
    return edge

def get_hole_index(depth):
    index = []
    for row in range(555):
        for col in range(690):
            if depth[row][col] == 0:
                index.append([row, col])
    print(index)
    return index

def generate_clean_edge(index, Ein ):
    # step 1
    G = cv2.getGaussianKernel((9,9), 1)
    I = np.zeros((555,690))
    E1 = np.zeros((555, 690))
    th = 0.555
    for [row, col] in index:
        I[row][col] = 1
    for i in range(555):
        for j in range(690):
            if Ein[i][j] == 255:
                Iq = np.zeros((3, 3))
                for i1 in range(i-1,i+2):
                    for j1 in range(j-1,j+2):
                        Iq[i1][j1] = I[i1][j1]
                # Iq is a binary mask

                if np.sum(np.dot(G, Iq)) < th:
                    E1[i][j] = 1

    # step 2
    h = 21




def test():
    projectPath = '/home/jianquan/PycharmProjects/ELG5163-Depth-Image-Denoising/'
    rawDataPath = projectPath + r'kinect-like/'

    testimg = cv2.imread(rawDataPath + 'colorart.bmp', 0)

    #plt.imshow(testimg, 'gray')
    testdepth = cv2.imread(rawDataPath + 'kinectart.png', 0)
    index = get_hole_index(testdepth)

    #RGBedge = my_canny(testimg, index)
    RGBedge = Color_edge_denerator(testimg)
    # test
    # RGBedge = np.zeros((555,690))
    # for [i,j] in index:
    #     RGBedge[i][j] = 255
    Depthedge = Depth_edge_generator(testdepth)




    plt.subplot(121)
    plt.imshow(RGBedge, cmap='gray')
    plt.title('RGB edge')
    plt.subplot(122)
    plt.imshow(Depthedge, cmap='gray')
    plt.title('Depthedge')
    plt.show()

test()
