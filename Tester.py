import cv2
import cv2.cv as cv
import numpy as np
import tesseract
import re
import json
import sys

 
#######   training part    ############### 
samples = np.loadtxt('generalsamples3.data',np.float32)
responses = np.loadtxt('generalresponses3.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

# samples = np.load('samples1.data.npy')
# samples = np.loadtxt('generalsamples1.data',np.float32)
# responses = np.load('responses1.data.npy')

# responses = np.loadtxt('generalresponses1.data',dtype=str)
# responses = responses.reshape((responses.size,1))
# print responses
# responses = np.array(responses,np.float32)
# responses = np.float32(responses)
# print responses
# model = cv2.KNearest()
# model.train(samples,responses)



img = cv2.imread('1.tif')
out = np.zeros(img.shape,np.uint8)
print img.shape
img1 = cv2.imread('1.tif',0) 					# Read the image as GRAYSCALE
#cv.NamedWindow('Image')
cv2.resizeWindow('Image',100,100)
#cv2.imshow('Original image',img1)
blur = cv2.GaussianBlur(img1,(5,5),0)	# Blur the image using GaussianBlur
ret3,th3 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	# changed this 0 to 100 ---- Create binary image using OTSU thresholding
imageinv =  255 - th3     				# Invert the image to have white text in black background
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(imageinv,kernel,iterations = 1) 	# Erode the image
dilation = cv2.dilate(erosion,kernel,iterations = 2)	# Dilate the image
dilation_copy = dilation.copy()
contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



samples =  np.empty((0,2500))
print samples
print samples.shape
print type(samples)
responses = []
keys = [i for i in range(44,123)]



for cnt in contours:
    if cv2.contourArea(cnt)>200:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>10:
			print 'hi inside h>50 if block'
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			roi = dilation[y:y+h,x:x+w]
			print type(roi),roi.shape
			roismall = cv2.resize(roi,(50,50))
			print type(roismall),roismall.shape
			roismall = roismall.reshape((1,2500))
			roismall = np.float32(roismall)
			retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
			string = str(int((results[0][0])))
			cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
			cv2.putText(img,string,(x,y+h),0,1,(0,255,0))
			cv2.imshow('Image',img)
			key = cv2.waitKey(0)

			

cv2.imshow('Image',img)
cv2.imwrite('OCR1.png',img)
cv2.imshow('out',out)
cv2.waitKey(0)

'''
count = 0
for cnt in contours:
	
#for i in range(0, len(contours)):
	if cv2.contourArea(cnt) > 500:
		count += 1
		leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
		rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
		topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
		bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
		
		rect = cv2.minAreaRect(cnt)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		
		# rect1 = cv2.boundingRect(cnt)
	# 	if rect1[2] < 100 or rect1[3] < 100: continue
		
		M = cv2.moments(cnt)
		centroid_x = int(M['m10']/M['m00'])
		centroid_y = int(M['m01']/M['m00'])
		
		print 'LeftMost: '+ str(count),(leftmost)
		print 'RightMost: ',(rightmost)
		print 'TopMost: ',(topmost)
		print 'Bottommost: ',(bottommost)
		
		print 'Centroid X: ',(centroid_x)
		print 'Centroid Y: ',(centroid_y)
		
		print 'Rect: ',(rect)
		print 'Box: ',(box)
		x,y,w,h = cv2.boundingRect(cnt)
		#x,y,w,h = rect
		cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
		letter = dilation_copy[y:y+h,x:x+w]
		cv2.imwrite('Z_'+str(count)+'.png', letter)

cv2.imshow('Output',img1)
cv2.imwrite('Contours.png',img1)
cv2.waitKey(5000)
cv2.destroyAllWindows()
'''