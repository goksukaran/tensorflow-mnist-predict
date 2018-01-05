'''
Created on Dec 8, 2017

@author: goksukara
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Hand_digit_Tensorflow_predict import imageprepare,predictint


im = cv2.imread('./Images/Ab_8.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY)
cv2.waitKey(0);
plt.imshow(imgray)
plt.show()
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


for i in range(1,len(contours)):

    x,y,w,h = cv2.boundingRect(contours[i])
#im3 = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    roi = im[y-10:y+h+10, x-10:x+w+10]
    imvalue =imageprepare(roi);
    predint = predictint(imvalue)
    cv2.imwrite('./Images/Numbers_Tensor_flow_predicted:'+str(predint)+'.png',roi)
    plt.imshow(roi)
    plt.title(predint)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
   
    plt.show()
