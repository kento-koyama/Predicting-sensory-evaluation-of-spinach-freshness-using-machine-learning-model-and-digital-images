
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
files = glob.glob("*")
a= np.arange(1,10)

for i in a:
    img = cv2.imread(files[i],0)
    img = img[0 : -1, 145: -145]
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.GaussianBlur(img,(3,3),0)
    edges = cv2.Canny(img,40,50)    
    #Original
    plt.subplot(331),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'+str(i)), plt.xticks([]), plt.yticks([])
    #Edge
    plt.subplot(332),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #ORB
    orb = cv2.ORB_create()    
    kp = orb.detect(img,None)    
    kp, des = orb.compute(img, kp)
    img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0), flags=0)
    plt.subplot(333),plt.imshow(img2)
    plt.title('ORB'), plt.xticks([]), plt.yticks([])
    #Sobel
    img_sobel1 = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    plt.subplot(334),plt.imshow(img_sobel1,cmap = 'gray')
    plt.title('Sobel x'), plt.xticks([]), plt.yticks([])
    #Sobel
    img_sobel2 = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    plt.subplot(335),plt.imshow(img_sobel2,cmap = 'gray')
    plt.title('Sobel y'), plt.xticks([]), plt.yticks([])
    #lap
    img_lap = cv2.Laplacian(img, cv2.CV_32F)
    plt.subplot(336),plt.imshow(img_lap,cmap = 'gray')
    plt.title('Lap y'), plt.xticks([]), plt.yticks([])
    #hist
    plt.subplot(337)
    plt.hist(img_sobel1.flatten())
    plt.subplot(338)
    plt.hist(img_sobel1.flatten())
    plt.subplot(339)
    plt.hist(img_lap.flatten())

    plt.show()    
