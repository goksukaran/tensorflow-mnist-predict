'''
Created on Dec 8, 2017

@author: goksukara
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Hand_digit_Tensorflow_predict import imageprepare,predictint
from PictureCapture import captureandsave







'''for i in range(1,len(contours)):

    x,y,w,h = cv2.boundingRect(contours[i])
#im3 = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    roi = thresh[y-10:y+h+10, x-10:x+w+10]
    imvalue =imageprepare(roi);
    predint = predictint(imvalue)
    cv2.imwrite('./Images/Numbers_Tensor_flow_predicted:'+str(predint)+'.png',roi)
    plt.imshow(roi,cmap='gray')
    plt.title(predint)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
   
    plt.show()'''

def Image_threshold(img):
    ret, thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plt.imshow(im2,cmap='gray')
    plt.title('Thresholded Image')
    plt.show()
    return thresh,contours
def ContourRevision(contours):
    print(len(contours))
    index_revised_contours_index=0
    revised_contours=list()
    for i in range(1,len(contours)):
        area = cv2.contourArea(contours[i])
        print(area)
        if(area>400):
            
            revised_contours.append(contours[i])
    
 
    
    print(len(revised_contours))            
    return contours            
    
def GuessNumbers(thresh,contours):
    for i in range(1,len(contours)):

        x,y,w,h = cv2.boundingRect(contours[i])
#im3 = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        imvalue =imageprepare(roi);
        predint = predictint(imvalue)
        cv2.imwrite('./Images/Numbers_Tensor_flow_predicted:'+str(predint)+'.png',roi)
        plt.subplot(round(np.sqrt(len(contours)))+1,round(np.sqrt(len(contours)))+1,i)
        plt.imshow(roi,cmap='gray')
        plt.title(predint)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
   
    plt.show()
def main():
    
    #im=captureandsave()
    im= cv2.imread('./Images/capture.png',0)
    thresh,contours=Image_threshold(im)
    resived_contours=ContourRevision(contours)
    GuessNumbers(thresh,resived_contours)
   

if __name__ == "__main__":
    main()