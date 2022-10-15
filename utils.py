'''The goal of this program is to visualize the bounding boxes and other elemens contained in the COCO annotations in order to better understand the dataset
dataset source https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset?resource=download '''

import numpy as np
from PIL import Image
import json
import os.path as osp
import matplotlib.pyplot as plt
import cv2 
import time
import torch
import os

#######
## extracting cards from dataset (same functions will be used to extract cards from images in real time)
#######

def find_contours(img_path, thresh):
    image = cv2.imread(img_path)
    image1 = image.copy()
    # convert the image to grayscale format
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    return contours

def isolate_resize_card(img_path,contours, new_size):

    image = cv2.imread(img_path)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) <=1:
        return 0
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    res = image[y:y+h,x:x+w]
    res_resized = cv2.resize(res, new_size)
    res_resized = np.swapaxes(res_resized,1,2)
    res_resized = np.swapaxes(res_resized,0,1)
    return res_resized

def labels_from_text(str):
    #10 is treated differently than the others because it is the only card that needs 3 characters to define in the json file
    if str[1] == "0":
        if str[2] == "H":
            return 10*4-3
        if str[2] == "D":
            return 10*4-2
        if str[2] == "S":
            return 10*4-1
        if str[2] == "C":
            return 10*4
    if str[1] == "O":
        return 53
    #on traite les cartes hautes car on ne pourra pas convertir le premier caractère en int
    if str[0] == "A":
        if str[1] == "H":
            return 1
        if str[1] == "D":
            return 2
        if str[1] == "S":
            return 3
        if str[1] == "C":
            return 4

    if str[0] == "K":
        if str[1] == "H":
            return 13*4-3
        if str[1] == "D":
            return 13*4-2
        if str[1] == "S":
            return 13*4-1
        if str[1] == "C":
            return 13*4

    if str[0] == "Q":
        if str[1] == "H":
            return 12*4-3
        if str[1] == "D":
            return 12*4-2
        if str[1] == "S":
            return 12*4-1
        if str[1] == "C":
            return 12*4

    if str[0] == "J":
        if str[1] == "H":
            return 11*4-3
        if str[1] == "D":
            return 11*4-2
        if str[1] == "S":
            return 11*4-1
        if str[1] == "C":
            return 11*4
        if str[1] == "O":
            return 53

    #all the other cases
    else:
        i = int(str[0])
        if str[1] == "H":
            return (i-1)*4-3
        if str[1] == "D":
            return (i-1)-2
        if str[1] == "S":
            return (i-1)-1
        if str[1] == "C":
            return (i-1)

def int_to_mat_label(lab):
    lab = labels_from_text(lab) 
    label = np.zeros(53) 
    label[lab-1] = 1  
    return label

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./savedModels', save_filename)
    torch.save(network.state_dict(), save_path)

def display(img_path, contours):
    image = cv2.imread(img_path)
    image1 = image.copy()
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 346,461)
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image1,(x,y),(x+w,y+h),(255,0,0),thickness = 4)
    cv2.drawContours(image=image1, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)
    cv2.imshow('image',image1)
    cv2.waitKey(0)



## test part]
if __name__ == "__main__":
    f = open("./annotation.json")
    
    json_dictionary = json.load(f)
    test_img = json_dictionary["images"][1808]["file_name"]
    img_directory = "./dataset/Images/Images"
    test_img_path = osp.join(img_directory,test_img)
    contours = find_contours(test_img_path,170)
    display(test_img_path, contours)

    f.close()
## 


