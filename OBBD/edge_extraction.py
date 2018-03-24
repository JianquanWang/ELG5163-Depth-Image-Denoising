# Object Boundary Based Denoising for Depth Images
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import queue

projectPath = '/home/jianquan/PycharmProjects/ELG5163-Depth-Image-Denoising/'
rawDataPath = projectPath + r'kinect-like/'

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
# SurrogateP is called when p is an edge pixel, because it's ambiguous that which p belong to
def SurrogateP(p, C):
    # p=(x,y,C[x][y][:])
    x = p[0], y = p[1]
    minest = 999999999
    (raw,col) = (0,0)
    for i in range(x-1,x+1):
        for j in range(y-1, y+1):
            if i!=x and j!=y:
                tmp = math.sqrt((C[x][y][0]-C[i][j][0])**2 + (C[x][y][1]-C[i][j][1])**2 + (C[x][y][2]-C[i][j][2])**2)
                if tmp < minest:
                    minest = tmp
                    (raw,col) = (i, j)
    return (raw, col, C[raw][col][:])

# intensity kernel
def fi(C, q, p):

#def DepthHollFilling(E, Din):
#    Q = queue.PriorityQueue()



def test():
    ColorImage = cv2.imread(rawDataPath+'colorart.bmp')  #uint8
    DepthMap = cv2.imread(rawDataPath+'kinectart.png', 0)
    ColorEdgeMap = cv2.Canny(cv2.cvtColor(ColorImage, cv2.COLOR_RGB2GRAY), 7, 9)
    DepthEdgeMap = cv2.Canny(DepthMap, 7, 9)
    #DepthDistanceTransformation =
test()
