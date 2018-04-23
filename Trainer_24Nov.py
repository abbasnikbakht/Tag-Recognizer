import cv2
import cv2.cv as cv
import numpy as np
import tesseract
import re
import json
import sys
from StringIO import StringIO

bin_n = 16
SZ = 50
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):

	# print 'Inside hog function'
# 	print type(img)
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	#print gx
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	#print gy
	mag, ang = cv2.cartToPolar(gx, gy)
	#print mag
	#print ang
	bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
	#print bins
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	#print hists
	hist = np.hstack(hists)     # hist is a 64 bit vector
	return hist

#116 -> HOGfeatures4

img = cv2.imread('128_clean.jpg')
print img.shape
img1 = cv2.imread('128_clean.jpg',0) 					# Read the image as GRAYSCALE
#cv.NamedWindow('Image')
# cv2.resizeWindow('Image',100,100)
# #cv2.imshow('Original image',img1)
# blur = cv2.GaussianBlur(img1,(5,5),0)	# Blur the image using GaussianBlur
# ret3,th3 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	# changed this 0 to 100 ---- Create binary image using OTSU thresholding
# imageinv =  255 - th3     				# Invert the image to have white text in black background
# kernel = np.ones((5,5),np.uint8)
# #erosion = cv2.erode(imageinv,kernel,iterations = 1) 	# Erode the image
# dilation = cv2.dilate(imageinv,kernel,iterations = 1)	# Dilate the image

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))	# Create CLAHE object
#clahe = cv2.createCLAHE()
cl1 = clahe.apply(img1)				# Apply CLAHE histogram equalization
#cv2.imwrite('clahe_histequal.jpg',cl1)
blur = cv2.GaussianBlur(cl1,(5,5),0)	# Gaussian Blur
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	#OTSU Thresholding
#cv2.imwrite('ThreshCLAHEEqualizedOTSU.jpg',otsu)
imageinv =  255 - otsu     				# Invert the image to have white text in black background
#cv2.imwrite('CLAHEInvertedImage.jpg',imageinv)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(imageinv,kernel,iterations = 1) 	# Erode the image
#cv2.imwrite('CLAHEErodedImage.jpg',erosion)
dilation = cv2.dilate(erosion,kernel,iterations = 1)	# Dilate the image
#cv2.imwrite('CLAHEDilatedImage.jpg',dilation)
dilation_copy = dilation.copy()
#imageinv_copy = imageinv.copy()


#dilation_copy = dilation.copy()
contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

samples = np.loadtxt('HOGfeatures9.data',np.float32)
responses = np.loadtxt('HOGresp9.data',np.float32)

# samples =  np.empty((0,2500))
# print samples
# print samples.shape
# print type(samples)
# responses = []
# hogData =  np.empty((0,64))
# print hogData
# print hogData.shape
# print type(hogData)

keys = [i for i in range(44,123)]

samples = list(samples)
responses = list(responses)

for cnt in contours:
    if cv2.contourArea(cnt)>500:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>10:
			print 'hi inside h>50 if block'
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			roi = dilation[y:y+h,x:x+w]
			print type(roi),roi.shape
			roismall = cv2.resize(roi,(50,50))
			print type(roismall),roismall.shape
			
			deskewed_image = deskew(roismall)
			hogdata = hog(deskewed_image)
			print hogdata
			
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
				#sample = roismall.reshape((1,2500))
				#samples = np.append(samples,sample,0)
				
				hogdata_reshaped = np.float32(hogdata).reshape(-1,64)
				samples = np.append(samples,hogdata_reshaped,0)


				
				

	
	
	
print 'HOGsamples :  \n',samples
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
np.savetxt('HOGfeatures10.data',samples)
#np.save('responses1.data',responses)
#np.savetxt('generalresponses1.data', responses, fmt="%s") 
np.savetxt('HOGresp10.data',responses)

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