### This is preprocessing document for DECNN

## gray image pre-processing
import os
import cv2
import numpy as np
import h5py

project_path = r'/home/jianquan/PycharmProjects/ELG5163-Depth-Image-Denoising/'



# intensity equalization
def step1_equalization(img):
    equ = cv2.equalizeHist(img)
    return equ


# bilateral filtering
def step2_bilateralFilter(img):
    blur = cv2.bilateralFilter(img,9,75,75)
    return blur


# edge extraction
def step3_edgeExtraction(img):
    edge = cv2.Canny(img, 20, 30)
    return edge


# watershed segmentation
def step4_watershed(img,color):
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers1 = cv2.connectedComponents(sure_fg)
    markers = markers1 + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    wartershed = color
    wartershed[markers == -1] = [255, 0, 0]
    return wartershed

def process_gray():
    for relpath, dirs, files in os.walk(project_path+'DeCNN/Depth_Enh/01_Middlebury_Dataset/'):
        for file in files:
            if file[14:25] == 'clean_color':
                full_path = os.path.join(project_path+'DeCNN/Depth_Enh/01_Middlebury_Dataset/', relpath, file)
                print(os.path.normpath(os.path.abspath(full_path)))
                img = cv2.imread(full_path, 0)
                color = cv2.imread(full_path)
                watershed = step4_watershed(img,color)
                cv2.imshow(watershed)


### h5py file generator

def h5pygen():
    X = []
    Y = []
    f = h5py.File(project_path + 'DeCNN/Depth.h5', 'w')
    f.create_dataset("X_train_orig", (28,370,427,1))
    f.create_dataset("Y_train_orig", (28,370,427,1))
    f.create_dataset("X_test_orig", (2,370,427,1))
    f.create_dataset("Y_test_orig", (2,370,427,1))
    for relpath, dirs, files in os.walk(project_path + 'DeCNN/Depth_Enh/01_Middlebury_Dataset/'):

        for file in files:
            full_path = os.path.join(project_path + 'DeCNN/Depth_Enh/01_Middlebury_Dataset/', relpath, file)
            if file[14:25] == 'noisy_depth':
                X.append(cv2.imread(full_path, 0))
            if file[14:26] == 'output_depth':
                Y.append(cv2.imread(full_path, 0))
    for i in range(30):
        if i < 28:
            np.append(f["X_train_orig"][i], X[i])
            np.append(f["Y_train_orig"][i], Y[i])
        else:
            np.append(f["X_test_orig"][i-28], X[i])
            np.append(f["Y_test_orig"][i-28], Y[i])

    # X_train = np.array(X_train, dtype=np.uint8)
    # Y_train = np.array(Y_train, dtype=np.uint8)
    # X_test = np.array(X_test, dtype=np.uint8)
    # Y_test = np.array(Y_test, dtype=np.uint8)


h5pygen()
