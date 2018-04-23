import cv2
import numpy as np

img = cv2.imread('117_cut.jpg')
gray_img = cv2.imread('116_cut.jpg',0)
cv2.imwrite('clahe_1.jpg',gray_img)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray_img)
cv2.imwrite('clahe_2.jpg',cl1)

blur = cv2.GaussianBlur(gray_img,(5,5),0)
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('ThreshnohistequalizationOTSU.jpg',otsu)

blur1 = cv2.GaussianBlur(cl1,(5,5),0)
ret1, otsu1 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('ThreshCLAHEEqualizedOTSU.jpg',otsu1)


#img = cv2.imread('wiki.jpg',0)
equ = cv2.equalizeHist(gray_img)
res = np.hstack((gray_img,equ)) #stacking images side-by-side
cv2.imwrite('HistEqualized.jpg',res)

blur2 = cv2.GaussianBlur(res,(5,5),0)
ret2, otsu2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('ThreshHIstWEqualizedOTSU.jpg',otsu2)



##########Best Histogram Equalization code##########

gray_img = cv2.imread('126.jpg',0)
cv2.imwrite('clahe_grayimage.jpg',gray_img)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#clahe = cv2.createCLAHE()
cl1 = clahe.apply(gray_img)
cv2.imwrite('clahe_histequal.jpg',cl1)

blur = cv2.GaussianBlur(gray_img,(5,5),0)
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('ThreshnohistequalizationOTSU.jpg',otsu)


blur1 = cv2.GaussianBlur(cl1,(5,5),0)
cv2.imwrite('Threshlurred.jpg',blur1)
ret1, otsu1 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('ThreshCLAHEEqualizedOTSU.jpg',otsu1)


equ = cv2.equalizeHist(gray_img)
res = np.hstack((gray_img,equ)) #stacking images side-by-side
cv2.imwrite('HistEqualized.jpg',res)

blur2 = cv2.GaussianBlur(res,(5,5),0)
ret2, otsu2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('ThreshHIstWEqualizedOTSU.jpg',otsu2)

