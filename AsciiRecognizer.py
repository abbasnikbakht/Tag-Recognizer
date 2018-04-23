import cv2
import cv2.cv as cv
import numpy as np
import tesseract
import re
import json
import sys
from StringIO import StringIO



img = cv2.imread('111_new.jpg')
print img.shape
img1 = cv2.imread('111_new.jpg',0) 					# Read the image as GRAYSCALE
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


# samples =  np.empty((0,2500))
# print samples
# print samples.shape
# print type(samples)
responses = []
keys = [i for i in range(44,123)]

# samples = list(samples)
# responses = list(responses)

for cnt in contours:
    if cv2.contourArea(cnt)>200:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>10:
			# print 'hi inside h>50 if block'
			# cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			# roi = dilation[y:y+h,x:x+w]
			# print type(roi),roi.shape
			# roismall = cv2.resize(roi,(50,50))
			# print type(roismall),roismall.shape
			cv2.imshow('Image',img)
			key = cv2.waitKey(0)

			if key == 27:  # (escape to quit)
				sys.exit()
			elif key in keys:
				print key
				print chr(key)
				#print int(chr(key))
				if key in range(48,58):
					print key
					responses.append(key)
					responses.append(chr(key))
					#responses.append(int(chr(key)))
					
				elif key in range(65,123):
					print key
					responses.append(key)
					responses.append(chr(key))
				
				elif key in range(44,48):
					print key
					responses.append(key)
					responses.append(chr(key))
					
				#print responses
				

np.savetxt('ascii.txt',responses,fmt="%s")
