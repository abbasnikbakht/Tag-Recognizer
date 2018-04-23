import cv2
import numpy as np

img = cv2.imread('115_cut.jpg')
# gray_img = cv2.imread('116_cut.jpg',0)
# cv2.imwrite('clahe_1.jpg',gray_img)
# 
# # create a CLAHE object (Arguments are optional).
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(gray_img)
# cv2.imwrite('clahe_2.jpg',cl1)
# 
# blur = cv2.GaussianBlur(gray_img,(5,5),0)
# ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imwrite('ThreshnohistequalizationOTSU.jpg',otsu)
# 
# blur1 = cv2.GaussianBlur(cl1,(5,5),0)
# ret1, otsu1 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imwrite('ThreshCLAHEEqualizedOTSU.jpg',otsu1)
# 
# 
# #img = cv2.imread('wiki.jpg',0)
# equ = cv2.equalizeHist(gray_img)
# res = np.hstack((gray_img,equ)) #stacking images side-by-side
# cv2.imwrite('HistEqualized.jpg',res)
# 
# blur2 = cv2.GaussianBlur(res,(5,5),0)
# ret2, otsu2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imwrite('ThreshHIstWEqualizedOTSU.jpg',otsu2)




#BLUE_MIN = np.array([90, 25, 35],np.uint8)
#BLUE_MAX = np.array([150, 255, 255],np.uint8)

BLUE_MIN = np.array([90, 25, 35],np.uint8)
BLUE_MAX = np.array([200, 255, 255],np.uint8)


hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imwrite('outputhsv.jpg',hsv_img)

frame_threshed = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)
cv2.imwrite('output2.jpg', frame_threshed)

img = cv2.imread('outputhsv.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0)
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('Thresh.jpg',otsu)


kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(frame_threshed,kernel,iterations = 1) 	# Erode the image
cv2.imwrite('oErodedImage.jpg',erosion)
dilation = cv2.dilate(erosion,kernel,iterations = 1)	# Dilate the image
cv2.imwrite('oDilatedImage.jpg',dilation)


#Histogram
# gray_img = cv2.imread('117_cut.jpg',0)
# hist,bins = np.histogram(gray_img,256,[0,256])
# hist(gray_img.ravel(),256,[0,256])
# cv2.imwrite('histogram.jpg',hist)
# cv2.imshow('Image',hist)
# 
# 
