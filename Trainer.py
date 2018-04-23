import cv2
import cv2.cv as cv
import numpy as np
import tesseract
import re
import json
import sys
from StringIO import StringIO


# def myArrayConverter(arr):

    # convertArr = []
    # for s in arr:    
        # try:
            # value = np.float32(s)
        # except ValueError:
            # value = s

        # convertArr.append(value)

    # return np.array(convertArr,dtype=object)



img = cv2.imread('113_new.jpg')
print img.shape
img1 = cv2.imread('113_new.jpg',0) 					# Read the image as GRAYSCALE
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

samples = np.loadtxt('generalsamples2.data',np.float32)
responses = np.loadtxt('generalresponses2.data',np.float32)

# samples =  np.empty((0,2500))
# print samples
# print samples.shape
# print type(samples)
# responses = []
keys = [i for i in range(44,123)]

samples = list(samples)
responses = list(responses)

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
					responses.append(int(key))
					#responses.append(int(chr(key)))
					
				elif key in range(65,123):
					print key
					responses.append(int(key))
					#responses.append(chr(key))
				
				elif key in range(44,48):
					print key
					responses.append(int(key))
					#responses.append(chr(key))
					
				#print responses
				print "hi"
				sample = roismall.reshape((1,2500))
				samples = np.append(samples,sample,0)


				
				

	
	
	
print 'samples :',samples
#responses = np.float32(responses)
print type(responses)
print responses
# x = np.array(responses, dtype='|S4')
# print x
#y=[]
# for a in list(x):
       # try:
         # y.append(float(a))
       # except:
         # y.append(a)
# y=np.array(y, dtype=object)
# try:
	# y = responses.astype(float)
# except ValueError:
	# print 'raised'
# finally:
	# print y
	#responses_float = myArrayConverter(responses)
	#responses = np.array(responses)
	#print responses
	#print responses_float
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
#responses = responses.reshape((responses_float.size,1))
#responses = y.reshape((y.size,1))
print responses
print "training complete"

#np.save('samples1.data',samples)
np.savetxt('generalsamples3.data',samples)
#np.save('responses1.data',responses)
#np.savetxt('generalresponses1.data', responses, fmt="%s") 
np.savetxt('generalresponses3.data',responses)

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