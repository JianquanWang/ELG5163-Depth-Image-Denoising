# Object Boundary Based Denoising for Depth Images
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os



def RGB_edge_generator(img):
    #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0.1)
    edge = cv2.Canny(img, 70, 100)
    return edge


def Depth_edge_generator(depth):
    medianFilted = cv2.medianBlur(depth, 5)
    edge = cv2.Canny(medianFilted, 10, 100)
    return edge



def test():
    projectPath = '/home/jianquan/PycharmProjects/ELG5163-Depth-Image-Denoising/'
    rawDataPath = projectPath + r'kinect-like/'

    testimg = cv2.imread(rawDataPath + 'colorart.bmp', 0)
    plt.imshow(testimg, 'gray')
    testdepth = cv2.imread(rawDataPath + 'kinectart.png', 0)

    RGBedge = RGB_edge_generator(testimg)
    Depthedge = Depth_edge_generator(testdepth)

    plt.subplot(121)
    plt.imshow(RGBedge, cmap='gray')
    plt.title('RGB edge')
    plt.subplot(122)
    plt.imshow(Depthedge, cmap='gray')
    plt.title('Depthedge')
    plt.show()

test()
