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
    f = h5py.File(project_path + 'DeCNN/train.h5', 'w')
    #f.create_dataset("train_set_x", (28,370,427,1))
    #f.create_dataset("train_set_y", (28,370,427,1))
    f1 = h5py.File(project_path + 'DeCNN/test.h5', 'w')
    #f1.create_dataset("test_set_x", (2,370,427,1))
    #f1.create_dataset("test_set_y", (2,370,427,1))

    X = []
    Y = []
    for relpath, dirs, files in os.walk(project_path + 'DeCNN/Depth_Enh/01_Middlebury_Dataset/'):
        for file in files:
            full_path = os.path.join(project_path + 'DeCNN/Depth_Enh/01_Middlebury_Dataset/', relpath, file)
            if file[14:25] == 'noisy_depth':
                im = cv2.imread(full_path, 0)
                im = im.tolist()[0:370]
                for i in range(len(im)):
                    im[i] = im[i][0:413]
                X.append(im)
            if file[14:26] == 'output_depth':
                im = cv2.imread(full_path, 0)
                im = im.tolist()[0:370]
                for i in range(len(im)):
                    im[i] = im[i][0:413]
                Y.append(im)
    X1 = X[28:30]
    Y1 = Y[28:30]
    X = X[0:28]
    Y = Y[0:28]
    X = np.array(X).reshape(28,370,413,1)
    Y = np.array(Y).reshape(28, 370, 413, 1)
    X1 = np.array(X1).reshape(2, 370, 413, 1)
    Y1 = np.array(Y1).reshape(2, 370, 413, 1)
    #Y = np.array(Y)
    #X = np.array(X)
    #Y = np.array(Y)
    f.create_dataset("train_set_x", data=X)
    f.create_dataset("train_set_y", data=Y)
    f1.create_dataset("test_set_x", data=X1)
    f1.create_dataset("test_set_y", data=Y1)
    # f["train_set_x"] = X
    # f["train_set_y"] = Y
    # f1["test_set_x"] = X1
    # f1["test_set_y"] = Y1
    # for i in range(30):
    #     if i < 28:
    #         np.append(f["train_set_x"][i], X[i])
    #         np.append(f["train_set_y"][i], Y[i])
    #         print(f["train_set_x"][i])
    #     else:
    #         np.append(f1["test_set_x"][i-28], X[i])
    #         np.append(f1["test_set_y"][i-28], Y[i])

    # X_train = np.array(X_train, dtype=np.uint8)
    # Y_train = np.array(Y_train, dtype=np.uint8)
    # X_test = np.array(X_test, dtype=np.uint8)
    # Y_test = np.array(Y_test, dtype=np.uint8)


h5pygen()

f = h5py.File(project_path+'DeCNN/train.h5', 'r')
x = np.array(f['train_set_x'][:])
print(x.shape)
print(x[0].shape)
print(x)