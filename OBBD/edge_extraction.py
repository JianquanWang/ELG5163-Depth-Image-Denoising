# Object Boundary Based Denoising for Depth Images
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import queue
import time
from PIL import Image
projectPath = '/home/jianquan/PycharmProjects/ELG5163-Depth-Image-Denoising/'
rawDataPath = projectPath + r'kinect-like/'
from numba import jit


def get_shape(image):
    rows, cols, channels = image.shape
    return rows, cols, channels

def split(image):
    *_, channels = image.get_shape()
    if channels == 3:
        color = True  # color
        b = image[:][:][0]
        g = image[:][:][1]
        r = image[:][:][2]
        return b,g,r, color
    else:
        color = False  # gray
        return color

def pixelByCoor(image, x, y):
    *_, channels = image.get_shape()
    if channels == 3:
        return (x, y, image[x][y][0], image[x][y][1], image[x][y][2])
    else:
        return (x, y, image[x][y])

def color2Gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def morph_close(image):
    kernel = np.ones((13,13), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def showImage(image):
    plt.imshow(image)
    plt.show()




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

def SurrogateP(p, C, Din):
    x = p[0]
    y = p[1]
    minest = 9999999999999
    raw = 0
    col = 0
    for i in range(x-1,x+1):
            for j in range(y-1, y+1):
                if (i!=x or j!=y) and i<C.shape[0] and j<C.shape[1]:
                    tmp = abs(int(C[x][y][0])-int(C[i][j][0])) + abs(int(C[x][y][1])-int(C[i][j][1])) + abs(int(C[x][y][2])-int(C[i][j][2]))
                    if tmp < minest:
                        minest = tmp
                        raw = i
                        col = j
    return (raw, col, Din[raw][col])

def isedge(p, edgeimage):
    #print("isedge called")
   # t1 = time.time()
    A = np.where(edgeimage==255)
    A = np.transpose(A)
    if [p[0],p[1]] in A:
        return True
    else:
        return False

# intensity kernel

def fi(C, q, p, E, Din):



    qx = q[0]
    qy = q[1]
    px = p[0]
    py = p[1]
    tmp = math.sqrt((int(C[qx][qy][0])-int(C[px][py][0]))**2 + (int(C[qx][qy][1])-int(C[px][py][1]))**2 + (int(C[qx][qy][2])-int(C[px][py][2]))**2)

    if tmp < 100:
        return 1
    return 0

# range kernel

def fr(E, q, p):
    qx = q[0]
    qy = q[1]
    px = p[0]
    py = p[1]
    e = abs(qx-px) - abs(qy-py)
    if e >= 0: # x long
        if qx>px:
            row = np.arange(px+1, qx, 0.5, dtype=np.float)
            col = np.linspace(py,qy,row.shape[0], dtype=np.int)
        else:
            row = np.arange(qx+1, px, 0.5, dtype=np.float)
            col = np.linspace(qy, py, row.shape[0], dtype=np.int)
    else:
        if qy>py:
            col = np.arange(py+1, qy, 0.5, dtype=np.float)
            row = np.linspace(px,qx,col.shape[0], dtype=np.int)
        else:
            col = np.arange(qy+1, py, 0.5, dtype=np.float)
            row = np.linspace(qx, px, col.shape[0], dtype=np.int)
    row.astype(int)
    B = np.array([row,col])
    B = np.transpose(B)

    for pixel in B:
         if pixel in E:
             return 0
    return 1

def fs(q, p, C):
    #print("fs called")
    #t1 = time.time()
    #Euclidean = (q[0]-p[0])**2 + (q[1]-p[1])**2
    qx = q[0]
    qy = q[1]
    px = p[0]
    py = p[1]
    Euclidean = (int(C[qx][qy][0])-int(C[px][py][0]))**2 + (int(C[qx][qy][1])-int(C[px][py][1]))**2 + (int(C[qx][qy][2])-int(C[px][py][2]))**2
    # zero mean spatial Gaussian kernel with a standard deviation 1
    #t2 = time.time()
    #print("fs done," + str((t2 - t1) / 1000000) + "s")
    return math.exp(-(Euclidean)/2)

def Drec_pixel(n, p, E, C, Din, D):
    #print("Drec pixel called")
    t1 = time.time()
    if [p[0],p[1]] in E:
        p_ = SurrogateP(p, C, Din)
    else:
        p_ = p
    kp = 0
    tmp = 0
    for i in range(int(p[0]-(n-1)/2), int(p[0]+(n+1)/2)):
        for j in range(int(p[1]-(n-1)/2), int(p[1]+(n+1)/2)):
            if 0 <= i < D.shape[0] and 0 <= j < D.shape[1]:
                if i != p[0] and j != p[1]:
                    q = (i, j, D[i][j])
                    if fi(C, q, p_, E, D) != 0:
                        if fr(E, q, p) != 0:
                            FS = fs(q, p, C)
                            tmp += D[i][j] * FS
                            kp += FS
    if tmp>0:
        pp = (p[0], p[1], tmp)
        print("out", tmp/kp)
        D[p[0]][p[1]] = np.uint8(tmp/kp)
        t2 = time.time()
        print("Drec pixel done," + str(t2 - t1) + "s")


        #exit("a point have not deal with")
    return D

def DepthHollFilling(E, Din, C):
    n = 7
    Q = queue.PriorityQueue()
    #build the Q
    A = np.where(Din == 0)
    A = np.transpose(A)
    for pixel in A:
        x = pixel[0]
        y = pixel[1]
        p = (x, y, 0)
        holeNum = 0
        if x < Din.shape[0]-(n-1)/2 and y < Din.shape[1]-(n-1)/2:
            for i in range(int(x-(n-1)/2),int(x+(n+1)/2)):
                for j in range(int(y-(n-1)/2),int(y+(n+1)/2)):
                    if Din[i][j] == 0:
                        holeNum += 1
            Q.put((holeNum, p))
    D = Din
    #index = index - 1
    print("initalize done, enter While loop.")
    while not Q.empty():
        Qsize1 = Q.qsize()
        print("Q size is", Q.qsize())
        tmp = Q.get()
        x, y, _ = tmp[1]
        Din[x][y] = 0
        num = n*n-tmp[0]

        if num/(n*n) < 0.8:
            n = n+2
            p = tmp[1]
            # holeNum = 0
            # if x < Din.shape[0] - (n - 1) / 2 and y < Din.shape[1] - (n - 1) / 2:
            #     for i in range(int(x - (n - 1) / 2), int(x + (n + 1) / 2)):
            #         for j in range(int(y - (n - 1) / 2), int(y + (n + 1) / 2)):
            #             if D[i][j] == 0:
            #                 holeNum += 1
            #     Q.put((holeNum, p))
            Q.put(tmp)

        else:
            p = tmp[1]
            D = Drec_pixel(n, p, E, C, Din, D)
            holeNum = 0
            p = (x,y,D[x][y])
            if D[x][y] == 0:
                n = n+2
                if x < Din.shape[0] - (n - 1) / 2 and y < Din.shape[1] - (n - 1) / 2:
                    for i in range(int(x - (n - 1) / 2), int(x + (n + 1) / 2)):
                        for j in range(int(y - (n - 1) / 2), int(y + (n + 1) / 2)):
                            if D[i][j] == 0:
                                holeNum += 1
                    Q.put((holeNum, p))
        Qsize2 = Q.qsize()
        if Qsize1 == Qsize2:
            print(num, n, num/(n*n), tmp)
            # plt.imshow(D)
            # plt.show()
        # else:
        #     n=7
    #print("check point is", D[check[1][0]][check[1][1]])
    return D

def MAE(D, Din, Dpure):
    D = D[0:(D.shape[0] - 7)][0:(D.shape[1] - 7)]
    Din = Din[0:(Din.shape[0] - 7)][0:(Din.shape[1] - 7)]
    Dpure = Dpure[0:(Dpure.shape[0] - 7)][0:(Dpure.shape[1] - 7)]
    A = np.where(Din == 0)
    A = np.transpose(A)
    B = np.where(Dpure == 0)
    B = np.transpose(B)
    N = 0
    result = 0
    for pixel in A:
        if pixel not in B:
            x = pixel[0]
            y = pixel[1]
            N += 1
            result += abs(int(D[x][y]) - int(Dpure[x][y]))
    result = result/N
    return result

def test():
    tic = time.time()
    ColorImage = cv2.imread(rawDataPath+'colorart.bmp')  #uint8
    DepthMap = cv2.imread(rawDataPath+'kinectart.png', 0)
    #DepthMap = cv2.imread(projectPath+'OBBD/depth.png',0)
    PureDepth = cv2.imread(rawDataPath+'originart.bmp', 0)
    E = cv2.Canny(PureDepth,17,19)
    plt.imshow(E)
    E = np.where(E == 255)
    E = np.transpose(E)
    D = DepthHollFilling(E, DepthMap, ColorImage)
    #print(D)
    plt.imshow(D)
    plt.show()
    toc = time.time()
    print((toc-tic), 's')
    mae = MAE(D, DepthMap, PureDepth)
    #D = np.array(D)

    D = Image.fromarray(D)
    D.show()
    print("mae=",mae)
    #plt.imshow(D,cmap='gray')


np.set_printoptions(threshold=np.inf)
test()
