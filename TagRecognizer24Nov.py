#!/opt/local/bin/python
import cv2
import cv2.cv as cv
import numpy as np
import sys
import tesseract
import re
import json
#from Testercall import OCR_train, OCR

# Make sure if you are receiving an image with tid and imgid





def tagPreprocessor_HSV(tid,image):
	#img = cv2.imread(image)		# Read the BGR image
	BLUE_MIN = np.array([90, 25, 35],np.uint8)		# HSV values for Blue_Min
	BLUE_MAX = np.array([200, 255, 255],np.uint8)	# HSV values for Blue_Max
	hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)	# Convert BGR to HSV color space
	cv2.imwrite('HSV_output.jpg',hsv_img)
	frame_threshed = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)	# Threshold using inRange
	cv2.imwrite('HSV_thresh.jpg', frame_threshed)
	#img = cv2.imread('outputhsv.jpg',0)
	blur = cv2.GaussianBlur(frame_threshed,(5,5),0)	# Gaussian Blur
	ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)		#OTSU thresholding
	cv2.imwrite('HSV_OTSUThresh.jpg',otsu)
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(frame_threshed,kernel,iterations = 1) 	# Erode the image
	cv2.imwrite('HSV_ErodedImage.jpg',erosion)
	dilation = cv2.dilate(erosion,kernel,iterations = 1)	# Dilate the image
	cv2.imwrite('HSV_DilatedImage.jpg',dilation)
	return dilation


def tagPreprocessor_HistEqualize(image) :     	
	img1 = cv2.imread(image,0) 					# Read the image as GRAYSCALE
	#cv.NamedWindow('Original image')
	cv2.imshow('Original image',img1)
	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))	# Create CLAHE object
	#clahe = cv2.createCLAHE()
	cl1 = clahe.apply(img1)				# Apply CLAHE histogram equalization
	cv2.imwrite('clahe_histequal.jpg',cl1)
	blur = cv2.GaussianBlur(cl1,(5,5),0)	# Gaussian Blur
	cv2.imwrite('clahe_blur.jpg',blur)

	# ret, otsu = cv2.threshold(cl1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	#OTSU Thresholding
# 	cv2.imwrite('CLAHEEqualizedOTSU.jpg',otsu)

	ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	#OTSU Thresholding
	cv2.imwrite('CLAHEEqualizedBlurredOTSU.jpg',otsu)


	imageinv =  255 - otsu     				# Invert the image to have white text in black background
	cv2.imwrite('CLAHEInvertedImage.jpg',imageinv)
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(imageinv,kernel,iterations = 1) 	# Erode the image
	cv2.imwrite('CLAHEErodedImage.jpg',erosion)
	dilation = cv2.dilate(erosion,kernel,iterations = 2)	# Dilate the image
	cv2.imwrite('CLAHEDilatedImage.jpg',dilation)
	return dilation #imageinv #erosion



def tagPreprocessor2(tid, imgid, image) :     	# tid is in range of 1,2,3,.... n template tags
	img1 = cv2.imread(image,0) 					# Read the image as GRAYSCALE
	#cv.NamedWindow('Original image')
	cv2.imshow('Original image',img1)
	#blur = cv2.GaussianBlur(img1,(5,5),0)	# Blur the image using GaussianBlur
	#cv2.imwrite('BlurredImage2.jpg',blur)
	#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	# Create binary image using OTSU thresholding
	#cv2.imwrite('ThresholdedImage2.jpg',th3)
	#imageinv =  255 - th3     				# Invert the image to have white text in black background
	#cv2.imwrite('InvertedImage2.jpg',imageinv)
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(img1,kernel,iterations = 1) 	# Erode the image
	dilation = cv2.dilate(erosion,kernel,iterations = 1)	# Dilate the image
	cv2.imwrite('DilatedImage2.jpg',dilation)	
	return dilation
	
def computeSkewAngle(dilation):
	dilation_copy = dilation.copy()
	contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	# Detect the contours of all objects
	leng = 0
	# for i in range(len(contours)):
# 		#print len(contours[i])
# 		leng = leng + len(contours[i])
# 
# 	print leng + 1

	points = []
	#pointMat = cv.CreateMat(leng, 1, cv.CV_32SC2)

	try:
		count = 0
		print 'Length of contours :', len(contours)
		print range(len(contours))
		for i in range(len(contours)):
			cnt = contours[i] 
			#print 'cnt :   ' , cnt    
			if cv2.contourArea(cnt) > 2000:	       #2000      #### check why it is going list out of index  OR EXCEPTION HANDLING ######
				print 'Found Biggggg contour in : ',i 	#print cnt
			if cv2.contourArea(cnt) < 500:				#changed from 500
				print 'Found Small Speckles in : ', i
			else:
				count = count + 1
				for i in range(len(cnt)):
					#print 'Length of each contour :   ',len(cnt)
					#pointMat[i, 0] = tuple(cnt[i][0])
					points.append(tuple(cnt[i][0]))
				

	except:
		print 'Exception raised'
		print "Unexpected error:", sys.exc_info()[0]
		
	finally:
		print 'COUNT :   ', count
		#print points
		print 'Hi'
		box = cv.MinAreaRect2(points)
		print box
		box_vtx = [roundxy(p) for p in cv.BoxPoints(box)]
		#box_vtx = [cv.Round(pt[0]), cv.Round(pt[1]) for p in cv.BoxPoints(box)]
		print box[2]
		if box[2] < -45:
			skew_angle = box[2] + 90
		else:
			skew_angle = box[2]
			

		print 'Skew Angle :  ',skew_angle
		print 'Box :   ', box_vtx
	return skew_angle,box_vtx	

def computeSkewAngle2(dilation):
	dilation_copy = dilation.copy()
	contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	# Detect the contours of all objects
	leng = 0
	for i in range(len(contours)):
		#print len(contours[i])
		leng = leng + len(contours[i])

	print leng + 1

	points = []
	pointMat = cv.CreateMat(leng, 1, cv.CV_32SC2)

	try:

		for i in range(len(contours[i])):
			cnt = contours[i]     
			if cv2.contourArea(contours[i]) > 10000:	             #### check why it is going list out of index  OR EXCEPTION HANDLING ######
				print 'Found Biggggg contour' #print cnt
			else:
				for i in range(len(cnt)):
					pointMat[i, 0] = tuple(cnt[i][0])
					points.append(tuple(cnt[i][0]))
				#print pointMat

	except:
		print 'Exception raised'
		print "Unexpected error:", sys.exc_info()[0]
		
	finally:
		print 'Hi'
		box = cv.MinAreaRect2(points)
		box_vtx = [roundxy(p) for p in cv.BoxPoints(box)]
		#box_vtx = [cv.Round(pt[0]), cv.Round(pt[1]) for p in cv.BoxPoints(box)]
		print box[2]
		if box[2] < -45:
			skew_angle = box[2] + 90
		else:
			skew_angle = box[2]
			

		print 'Skew Angle :  ',skew_angle
	return skew_angle,box_vtx	


def computeSkewAngle3(dilation):
	dilation_copy = dilation.copy()
	contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	# Detect the contours of all objects
	leng = 0
	for i in range(len(contours)):
		#print len(contours[i])
		leng = leng + len(contours[i])

	print leng + 1

	points = []
	pointMat = cv.CreateMat(leng, 1, cv.CV_32SC2)

	try:

		for i in range(len(contours[i])):
			cnt = contours[i]     
			if cv2.contourArea(contours[i]) > 10000:	             #### check why it is going list out of index  OR EXCEPTION HANDLING ######
				print 'Found Biggggg contour' #print cnt
			else:
				for i in range(len(cnt)):
					pointMat[i, 0] = tuple(cnt[i][0])
					points.append(tuple(cnt[i][0]))
				#print pointMat

	except:
		print 'Exception raised'
		print "Unexpected error:", sys.exc_info()[0]
		
	finally:
		print 'Hi'
		box = cv.MinAreaRect2(points)
		box_vtx = [roundxy(p) for p in cv.BoxPoints(box)]
		#box_vtx = [cv.Round(pt[0]), cv.Round(pt[1]) for p in cv.BoxPoints(box)]
		print box[2]
		if box[2] < -45:
			skew_angle = box[2] + 90
		else:
			skew_angle = box[2]
			

		print 'Skew Angle :  ',skew_angle
	return skew_angle,box_vtx	


def computeSkewAngle4(dilation):
	dilation_copy = dilation.copy()
	contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	# Detect the contours of all objects
	leng = 0
	for i in range(len(contours)):
		#print len(contours[i])
		leng = leng + len(contours[i])

	print leng + 1

	points = []
	pointMat = cv.CreateMat(leng, 1, cv.CV_32SC2)

	try:

		for i in range(len(contours[i])):
			cnt = contours[i]     
			if cv2.contourArea(contours[i]) > 10000:	             #### check why it is going list out of index  OR EXCEPTION HANDLING ######
				print 'Found Biggggg contour' #print cnt
			else:
				for i in range(len(cnt)):
					pointMat[i, 0] = tuple(cnt[i][0])
					points.append(tuple(cnt[i][0]))
				#print pointMat

	except:
		print 'Exception raised'
		print "Unexpected error:", sys.exc_info()[0]
		
	finally:
		print 'Hi'
		box = cv.MinAreaRect2(points)
		box_vtx = [roundxy(p) for p in cv.BoxPoints(box)]
		#box_vtx = [cv.Round(pt[0]), cv.Round(pt[1]) for p in cv.BoxPoints(box)]
		print box[2]
		if box[2] < -45:
			skew_angle = box[2] + 90
		else:
			skew_angle = box[2]
			

		print 'Skew Angle :  ',skew_angle
	return skew_angle,box_vtx	


def rotateImage(skew_angle,box_vtx,dilation):
	#image_copy2 = image.copy()
	dilation_copy2 = dilation.copy()
	#image = 'Z_Cropped_color_image.png'
	ori_image = cv2.imread(image)

	rows,cols,ch = ori_image.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),skew_angle,1)
	dst = cv2.warpAffine(dilation_copy2,M,(cols,rows))
	rot_img_string = 'Z_Rotated_image.jpg'
	cv2.imwrite('Z_Rotated_image.jpg',dst)
	dst1 = cv2.warpAffine(ori_image,M,(cols,rows))
	cv2.imwrite('Z_Rotated_color_image.jpg',dst1)
	#b = cv.CreateMat(erosion.shape[0], erosion.shape[1], cv.CV_8UC1)
	#b = cv.fromarray(erosion)

	#cv.PolyLine(b, [box_vtx], 1, cv.CV_RGB(0, 255, 255), 1, cv.CV_AA)
	#cv2.polylines(erosion,[box_vtx],True,(255,0,0),2)
	cv2.polylines(ori_image,np.array([box_vtx],dtype=np.int32),True,(255,0,0),2)

	#erosioncopy = erosion[530:603,992:1037]
	#image = np.asarray(b[:,:])
	#cv2.imshow('Boxed image',erosion)
	cv2.imwrite('Z_Boxed_image.jpg',ori_image)
	print 'Out of rotateImage function'
	return dst, dst1,rot_img_string
		

def rotateImage2(skew_angle,box_vtx,dilation):
	#image_copy2 = image.copy()
	dilation_copy2 = dilation.copy()
	#image = 'Z_Cropped_color_image.png'
	ori_image = cv2.imread(image)

	rows,cols,ch = ori_image.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),skew_angle,1)
	dst = cv2.warpAffine(dilation,M,(cols,rows))
	rot_img_string = 'Z_Rotated_image2.jpg'
	cv2.imwrite('Z_Rotated_image2.jpg',dst)
	dst1 = cv2.warpAffine(ori_image,M,(cols,rows))
	cv2.imwrite('Z_Rotated_color_image2.jpg',dst1)
	#b = cv.CreateMat(erosion.shape[0], erosion.shape[1], cv.CV_8UC1)
	#b = cv.fromarray(erosion)

	#cv.PolyLine(b, [box_vtx], 1, cv.CV_RGB(0, 255, 255), 1, cv.CV_AA)
	#cv2.polylines(erosion,[box_vtx],True,(255,0,0),2)
	cv2.polylines(ori_image,np.array([box_vtx],dtype=np.int32),True,(255,0,0),2)

	#erosioncopy = erosion[530:603,992:1037]
	#image = np.asarray(b[:,:])
	#cv2.imshow('Boxed image',erosion)
	cv2.imwrite('Z_Boxed_image2.jpg',ori_image)
	print 'Out of rotateImage2 function'
	return dst, rot_img_string
		
def rotateImage3(skew_angle,box_vtx,dilation):
	#image_copy2 = image.copy()
	dilation_copy2 = dilation.copy()
	#image = 'Z_Cropped_color_image.png'
	ori_image = cv2.imread(image)

	rows,cols,ch = ori_image.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),skew_angle,1)
	dst = cv2.warpAffine(dilation,M,(cols,rows))
	rot_img_string = 'Z_Rotated_image3.jpg'
	cv2.imwrite('Z_Rotated_image3.jpg',dst)
	dst1 = cv2.warpAffine(ori_image,M,(cols,rows))
	cv2.imwrite('Z_Rotated_color_image3.jpg',dst1)
	#b = cv.CreateMat(erosion.shape[0], erosion.shape[1], cv.CV_8UC1)
	#b = cv.fromarray(erosion)

	#cv.PolyLine(b, [box_vtx], 1, cv.CV_RGB(0, 255, 255), 1, cv.CV_AA)
	#cv2.polylines(erosion,[box_vtx],True,(255,0,0),2)
	cv2.polylines(ori_image,np.array([box_vtx],dtype=np.int32),True,(255,0,0),2)

	#erosioncopy = erosion[530:603,992:1037]
	#image = np.asarray(b[:,:])
	#cv2.imshow('Boxed image',erosion)
	cv2.imwrite('Z_Boxed_image3.jpg',ori_image)
	print 'Out of rotateImage3 function'
	return dst, rot_img_string
	
	

def rotateImage4(skew_angle,box_vtx,dilation):
	#image_copy2 = image.copy()
	dilation_copy2 = dilation.copy()
	#image = 'Z_Cropped_color_image.png'
	ori_image = cv2.imread(image)

	rows,cols,ch = ori_image.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),skew_angle,1)
	dst = cv2.warpAffine(dilation_copy2,M,(cols,rows))
	rot_img_string = 'Z_Rotated_image4.jpg'
	cv2.imwrite('Z_Rotated_image4.jpg',dst)
	dst1 = cv2.warpAffine(ori_image,M,(cols,rows))
	cv2.imwrite('Z_Rotated_color_image4.jpg',dst1)
	#b = cv.CreateMat(erosion.shape[0], erosion.shape[1], cv.CV_8UC1)
	#b = cv.fromarray(erosion)

	#cv.PolyLine(b, [box_vtx], 1, cv.CV_RGB(0, 255, 255), 1, cv.CV_AA)
	#cv2.polylines(erosion,[box_vtx],True,(255,0,0),2)
	cv2.polylines(ori_image,np.array([box_vtx],dtype=np.int32),True,(255,0,0),2)

	#erosioncopy = erosion[530:603,992:1037]
	#image = np.asarray(b[:,:])
	#cv2.imshow('Boxed image',erosion)
	cv2.imwrite('Z_Boxed_image4.jpg',ori_image)
	print 'Out of rotateImage4 function'
	return dst, rot_img_string
	
def findExtremePoints(rot_image,rot_img_string):
	print type(rot_image)
	image_copy = rot_image.copy()
	#image1 = tagPreprocessor(111,8,rot_img_string)
	# image1 = cv2.imread(rot_img_string)
	# print type(image1)
	lmost = []
	lmost1 = []
	rmost = []
	tmost = []
	bmost = []
	blob_count = 0
	area_list = []
	#image1 = np.asarray(image1[:,:])
	cv2.imshow('Rot image',image_copy)
	contours,hierarchy = cv2.findContours(image_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	# Detect the contours of all objects
	
	
	for i in range(len(contours)):
		print 'len of',i+1,'th contour :',len(contours[i]),'Area of',i+1,'th contour : ',cv2.contourArea(contours[i])
		#print 'Area of', i+1, 'th contour :', cv2.contourArea(contours[i])
		#area_list[i] = cv2.contourArea(contours[i])
		# if cv2.contourArea(contours[i]) > 5000:
# 			print 'FOund Big Contour > 10000 size'
# 		if cv2.contourArea(contours[i]) < 100:
# 			print 'FOund Small Contour < 100 size'
# 		else: #cv2.contourArea(contours[i]) > 000:						# Discarding the small blobs
		
		
		if cv2.contourArea(contours[i]) > 600:		#changed from 600				# Discarding the small blobs
			#print 'Inside >300'
			if cv2.contourArea(contours[i]) > 2000:
				#print 'Inside > 2000'
				print 'Found Big Contour > 3000 size'
			else:
				#print 'HIIIIIIII'
				blob_count = blob_count + 1
				area_list.append(cv2.contourArea(contours[i]))
				lmost1 = tuple(contours[i][contours[i][:,:,0].argmin()][0])		# Leftmost point
				#print lmost1
				lmost.append(lmost1)											
				#print lmost
				rmost1 = tuple(contours[i][contours[i][:,:,0].argmax()][0])		# Rightmost point
				#print rmost1
				rmost.append(rmost1)
				#print rmost
				tmost1 = tuple(contours[i][contours[i][:,:,1].argmin()][0])		# Topmost point
				#print tmost1
				tmost.append(tmost1)
				#print tmost
				bmost1 = tuple(contours[i][contours[i][:,:,1].argmax()][0])		# Bottommost point
				#print bmost1
				bmost.append(bmost1)
		
		# print 'lmost',lmost
		# print 'rmost',rmost
		# print 'tmost',tmost
		# print 'bmost',bmost	
		
		
		
		
		
	xl=[x[0] for x in lmost]						
	yt=[y[1] for y in tmost]
	xmin = min(xl)								# Finding xmin as minimum of x co-ordinates in Leftmost Point List
	ymin = min(yt)								# Finding ymin as minimum of y co-ordinates in Topmost Point list
	xr=[x[0] for x in rmost]
	yb=[y[1] for y in bmost]
	xmax = max(xr)								# Finding xmax as maximum of x co-ordinates in Righttmost Point List
	ymax = max(yb)								# Finding ymax as maximum of y co-ordinates in Bottommost Point List
	
	print 'Out of FindExtremepoints function'
	return xmin,xmax,ymin,ymax


def imgCropper1(xmin,xmax,ymin,ymax,image_to_be_cropped,color_image_to_be_cropped):
	print type(image_to_be_cropped)
	image = np.asarray(image_to_be_cropped[:,:])
	print 'after conversion hihihihii',type(image)
	b = cv.CreateMat(image_to_be_cropped.shape[0], image_to_be_cropped.shape[1], cv.CV_8UC1)
	b = cv.fromarray(image_to_be_cropped)
	# image_to_be_resized = cv.fromArray(image_to_be_resized)

	cv.SaveImage('before_cropping.jpg',b)
	#img = cv2.imread('Z_Rotated_color_image.jpg')
	crop_color_image = color_image_to_be_cropped[ymin:ymax,xmin:xmax]       # changed here
	crop_image = image_to_be_cropped[ymin:ymax,xmin:xmax]          # changed here
	crop_string = 'Z_Cropped_image'
	crop_img_string = crop_string + '.png'
	cv2.imwrite('Z_Cropped_color_image.png',crop_color_image)
	cv2.imwrite('Z_Cropped_image.png',crop_image)
	return crop_img_string, crop_image, crop_color_image

def imgCropper2(xmin,xmax,ymin,ymax,image_to_be_resized):
	print type(image_to_be_resized)
	image = np.asarray(image_to_be_resized[:,:])
	print 'after conversion hihihihii',type(image)
	b = cv.CreateMat(image_to_be_resized.shape[0], image_to_be_resized.shape[1], cv.CV_8UC1)
	b = cv.fromarray(image_to_be_resized)
	# image_to_be_resized = cv.fromArray(image_to_be_resized)

	cv.SaveImage('before_cropping2.jpg',b)
	img = cv2.imread('Z_Rotated_color_image2.jpg')
	crop_color_img = img[ymin:ymax,xmin:xmax]       # changed here
	cropppped = image[ymin:ymax,xmin:xmax]          # changed here
	crop_string = 'Z_Cropped_image2'
	crop_img_string = crop_string + '.png'
	cv2.imwrite('Z_Cropped_color_image2.png',crop_color_img)
	cv2.imwrite('Z_Cropped_image2.png',cropppped)
	return crop_img_string, cropppped, crop_color_img
	
def imgCropper3(xmin,xmax,ymin,ymax,image_to_be_resized):
	print type(image_to_be_resized)
	image = np.asarray(image_to_be_resized[:,:])
	print 'after conversion hihihihii',type(image)
	b = cv.CreateMat(image_to_be_resized.shape[0], image_to_be_resized.shape[1], cv.CV_8UC1)
	b = cv.fromarray(image_to_be_resized)
	# image_to_be_resized = cv.fromArray(image_to_be_resized)

	cv.SaveImage('before_cropping3.jpg',b)
	img = cv2.imread('Z_Rotated_color_image3.jpg')
	crop_color_img = img[ymin:ymax,xmin:xmax]       # changed here
	cropppped = image[ymin:ymax,xmin:xmax]          # changed here
	crop_string = 'Z_Cropped_image3'
	crop_img_string = crop_string + '.png'
	cv2.imwrite('Z_Cropped_color_image3.png',crop_color_img)
	cv2.imwrite('Z_Cropped_image3.png',cropppped)
	return crop_img_string, cropppped, crop_color_img



def imgCropper4(xmin,xmax,ymin,ymax,image_to_be_resized):
	print type(image_to_be_resized)
	image = np.asarray(image_to_be_resized[:,:])
	print 'after conversion hihihihii',type(image)
	b = cv.CreateMat(image_to_be_resized.shape[0], image_to_be_resized.shape[1], cv.CV_8UC1)
	b = cv.fromarray(image_to_be_resized)
	# image_to_be_resized = cv.fromArray(image_to_be_resized)

	cv.SaveImage('before_cropping4.jpg',b)
	img = cv2.imread('Z_Rotated_color_image4.jpg')
	crop_color_img = img[ymin:ymax,xmin:xmax]       # changed here
	cropppped = image[ymin:ymax,xmin:xmax]          # changed here
	crop_string = 'Z_Cropped_image4'
	crop_img_string = crop_string + '.png'
	cv2.imwrite('Z_Cropped_color_image4.png',crop_color_img)
	cv2.imwrite('Z_Cropped_image4.png',cropppped)
	return crop_img_string, cropppped, crop_color_img


def imgResizer1(crop_img_string, width, height):
	image1 = cv.LoadImage(crop_img_string,cv.CV_LOAD_IMAGE_GRAYSCALE)
	dst1 = cv.CreateImage((width,height), 8, 1)
	cv.Resize(image1,dst1,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_image.png', dst1)
	return dst1
	
def imgResizer(crop_img_string, width, height,crop_color_image):
	image1 = cv.LoadImage(crop_img_string,cv.CV_LOAD_IMAGE_GRAYSCALE)
	dstn1 = cv.CreateImage((width,height), 8, 1)
	cv.Resize(image1,dstn1,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_image.png', dstn1)
	# dst0 = cv.CreateImage((width,height), 8, 3)
# 	dst0 = cv2.resize(crop_color_img,(width,height))
	image2 = cv.LoadImage('Z_Cropped_color_image.png')
	dstn2 = cv.CreateImage((width,height), 8, 3)
	cv.Resize(image2,dstn2,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_color_image.png', dstn2)
	resized_color_image_string = 'Z_Resized_color_image.png'
	return dstn1,dstn2,resized_color_image_string
	

def imgResizer2(crop_img_string, width, height,crop_color_img):
	image1 = cv.LoadImage(crop_img_string,cv.CV_LOAD_IMAGE_GRAYSCALE)
	dst1 = cv.CreateImage((width,height), 8, 1)
	cv.Resize(image1,dst1,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_image2.png', dst1)
	# dst0 = cv.CreateImage((width,height), 8, 3)
# 	dst0 = cv2.resize(crop_color_img,(width,height))
	image2 = cv.LoadImage('Z_Cropped_color_image2.png')
	dst2 = cv.CreateImage((width,height), 8, 3)
	cv.Resize(image2,dst2,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_color_image2.png', dst2)
	return dst1,dst2

def imgResizer3(crop_img_string, width, height,crop_color_img):
	image1 = cv.LoadImage(crop_img_string,cv.CV_LOAD_IMAGE_GRAYSCALE)
	dst1 = cv.CreateImage((width,height), 8, 1)
	cv.Resize(image1,dst1,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_image3.png', dst1)
	# dst0 = cv.CreateImage((width,height), 8, 3)
# 	dst0 = cv2.resize(crop_color_img,(width,height))
	image2 = cv.LoadImage('Z_Cropped_color_image3.png')
	dst2 = cv.CreateImage((width,height), 8, 3)
	cv.Resize(image2,dst2,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_color_image3.png', dst2)
	return dst1,dst2


def imgResizer4(crop_img_string, width, height,crop_color_img):
	image1 = cv.LoadImage(crop_img_string,cv.CV_LOAD_IMAGE_GRAYSCALE)
	dst1 = cv.CreateImage((width,height), 8, 1)
	cv.Resize(image1,dst1,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_image4.png', dst1)
	# dst0 = cv.CreateImage((width,height), 8, 3)
# 	dst0 = cv2.resize(crop_color_img,(width,height))
	image2 = cv.LoadImage('Z_Cropped_color_image4.png')
	dst2 = cv.CreateImage((width,height), 8, 3)
	cv.Resize(image2,dst2,interpolation=cv.CV_INTER_LINEAR)
	cv.SaveImage('Z_Resized_color_image4.png', dst2)
	return dst1,dst2

def Cropper1(TagTemplates):
	croplist = []
	for i in range(TagTemplates[0][1]):
		croplist.append(imgCropper(TagTemplates[0][i+2][1],TagTemplates[0][i+2][2],TagTemplates[0][i+2][3],TagTemplates[0][i+2][4],resized,TagTemplates[0][i+2][0]))
		#print croplist
	return croplist


def Cropper(TagTemplates,tid):
	OCRCharlist = []
	##### matching tid with picking which template to choose for cropping placeholders ######
	for index in range(len(TagTemplates)):
		if(TagTemplates[index][0] == tid):
			print 'Found tid on index : ' ,index
			break
        else:
        	print 'tid Not found' 
        
	for i in range(TagTemplates[index][1]):
		print TagTemplates[index][1]
		crop_string,cropped = imgCropper(TagTemplates[index][i+2][1],TagTemplates[index][i+2][2],TagTemplates[index][i+2][3],TagTemplates[index][i+2][4],resized_dilation,TagTemplates[index][i+2][0])
		#crop_color_string,cropped_color = imgCropper(TagTemplates[0][i+2][1],TagTemplates[0][i+2][2],TagTemplates[0][i+2][3],TagTemplates[0][i+2][4],resized_color,TagTemplates[0][i+2][0])
		print crop_string, type(cropped)
		print 'HIIII'

		#OCRlist = tagRecognizer(croplist)    
		
		sortContoursList = sortContours(crop_string,cropped)		# STEP 9 : sort the contours in every cropped placeholder w.r.t x position
		#OCRlist = OCR(crop_string,cropped,crop_color_string,cropped_color)         # STEP 9 : call OCR(croplist[i]) for recognition of every placeholder
		OCRAsciilist = OCR_new(sortContoursList,cropped)          # STEP 10 : Recognize every character present in sortContoursList of every placeholder
		print 'OCRlist is :', OCRAsciilist
		
		
		Charlist = AsciiMapper(OCRAsciilist)		# STEP 11 : Map the Ascii characters in OCR output into actual characters
		print Charlist
		
		OCRCharlist.append(Charlist)
#	 	#newOCRlist = cleanOCR(OCRCharlist)   # STEP 12 : clean OCR output
# 		print newOCRlist
# 	
# 	
# 		json_string = JSONBuilder(newOCRlist, TagTemplates)   # STEP 13 : Build a JSON object
# 		print 'JSON object is :     '
# 		print json_string
	return OCRCharlist, index

def imgCropper(xmin,xmax,ymin,ymax,image_to_be_resized,crop_string):
	print 'Inside imgCropper' , type(image_to_be_resized)
	image = np.asarray(image_to_be_resized[:,:])
	print 'after conversion',type(image)
	# b = cv.CreateMat(image_to_be_resized.shape[0], image_to_be_resized.shape[1], cv.CV_8UC1)
# 	b = cv.fromarray(image_to_be_resized)
	#image_to_be_resized = cv.fromArray(image_to_be_resized)

	#cv.SaveImage('before_cropping.jpg',image_to_be_resized)
	cropppped = image[ymin:ymax+10,xmin:xmax+10]
	crop_img_string = crop_string + '.png'
	cv2.imwrite(crop_img_string,cropppped)
	return crop_img_string, cropppped


def OCR_train():
	#######   training part    ############### 
	# samples = np.loadtxt('generalsamples3.data',np.float32)
	# responses = np.loadtxt('generalresponses3.data',np.float32)
	# responses = responses.reshape((responses.size,1))

	# model = cv2.KNearest()
	# model.train(samples,responses)
	print 'Training Completed'
	return 0


def sortContours(croplist,cropped):
	OCRlist = []
	#print OCRlist
	print 'Croplist is :',croplist
	# for i in range(len(croplist)):
	#print len(croplist)
	#print croplist[i]
	print 'Inside OCR function'
	#image1 = tagPreprocessor(111,8,croplist[i])
	#image1=cv.LoadImage(croplist[i])
	#out = np.zeros(image1.shape,np.uint8)
	#print type(image1)
	#image1 = np.asarray(image1[:,:])
	#print type(image1)
	image1_copy = cropped.copy()
	#print type(image1_copy)
	contours,hierarchy = cv2.findContours(image1_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	print ' Hi after contour finding function call'
	
	contoursList = []
	for cnt in contours:
		if cv2.contourArea(cnt)>1500: 				#changed from 2500
			[x,y,w,h] = cv2.boundingRect(cnt)
			contoursList.append([x,y,w,h])
			print x,y,w,h
	
	print contoursList
	
	# sorting contours happens here
	sortContoursList = []
	sortContoursList = sorted(contoursList, key=lambda contours: contours[0])
	print sortContoursList
	return sortContoursList


def OCR_old(croplist,cropped):
	
	#dilation_copy = dilation.copy()

	

	# img = cv2.imread(image_file)
	
	# print img.shape
	# img1 = cv2.imread(image_file,0) 					# Read the image as GRAYSCALE
	# cv.NamedWindow('Image')
	# cv2.resizeWindow('Image',100,100)
	# cv2.imshow('Original image',img1)
	# blur = cv2.GaussianBlur(img1,(5,5),0)	# Blur the image using GaussianBlur
	# ret3,th3 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	# changed this 0 to 100 ---- Create binary image using OTSU thresholding
	# imageinv =  255 - th3     				# Invert the image to have white text in black background
	# kernel = np.ones((5,5),np.uint8)
	# erosion = cv2.erode(imageinv,kernel,iterations = 1) 	# Erode the image
	# dilation = cv2.dilate(erosion,kernel,iterations = 2)	# Dilate the image
	# dilation_copy = dilation.copy()
	# contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	OCRlist = []
	#print OCRlist
	print 'Croplist is :',croplist
	# for i in range(len(croplist)):
	#print len(croplist)
	#print croplist[i]
	print 'Inside OCR function'
	#image1 = tagPreprocessor(111,8,croplist[i])
	#image1=cv.LoadImage(croplist[i])
	#out = np.zeros(image1.shape,np.uint8)
	#print type(image1)
	#image1 = np.asarray(image1[:,:])
	#print type(image1)
	image1_copy = cropped.copy()
	#print type(image1_copy)
	contours,hierarchy = cv2.findContours(cropped,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	print ' Hi after contour finding function call'
	
	for cnt in contours:
		if cv2.contourArea(cnt)>200:
			[x,y,w,h] = cv2.boundingRect(cnt)

			if  h>10:
				print 'hi inside h>50 if block'
				cv2.rectangle(image1_copy,(x,y),(x+w,y+h),(0,0,255),2)
				roi = image1_copy[y:y+h,x:x+w]
				print type(roi),roi.shape
				roismall = cv2.resize(roi,(50,50))
				print type(roismall),roismall.shape
				roismall = roismall.reshape((1,2500))
				roismall = np.float32(roismall)
				retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
				string = str(int((results[0][0])))
				#cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
				print 'Output String is : ', string
				cv2.putText(image1_copy,string,(x,y+h),0,1,(0,255,0))
				#cv2.imshow('Image',image1)
				
				OCRlist.append(string)
				#key = cv2.waitKey(0)

				

	cv2.imshow('Image',image1_copy)
	cv2.imwrite('OCR2.png',image1_copy)
	#cv2.imshow('out',out)
	cv2.waitKey(0)
	return OCRlist

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

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
    
    
def OCR_new(sortContoursList,cropped):
	
	#dilation_copy = dilation.copy()

	

	# img = cv2.imread(image_file)
	
	# print img.shape
	# img1 = cv2.imread(image_file,0) 					# Read the image as GRAYSCALE
	# cv.NamedWindow('Image')
	# cv2.resizeWindow('Image',100,100)
	# cv2.imshow('Original image',img1)
	# blur = cv2.GaussianBlur(img1,(5,5),0)	# Blur the image using GaussianBlur
	# ret3,th3 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	# changed this 0 to 100 ---- Create binary image using OTSU thresholding
	# imageinv =  255 - th3     				# Invert the image to have white text in black background
	# kernel = np.ones((5,5),np.uint8)
	# erosion = cv2.erode(imageinv,kernel,iterations = 1) 	# Erode the image
	# dilation = cv2.dilate(erosion,kernel,iterations = 2)	# Dilate the image
	# dilation_copy = dilation.copy()
	# contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	OCRlist = []
	#print OCRlist
	#print 'Croplist is :',croplist
	# for i in range(len(croplist)):
	#print len(croplist)
	#print croplist[i]
	print 'Inside OCR_new function'
	#image1 = tagPreprocessor(111,8,croplist[i])
	#image1=cv.LoadImage(croplist[i])
	#out = np.zeros(image1.shape,np.uint8)
	#print type(image1)
	#image1 = np.asarray(image1[:,:])
	#print type(image1)
	image1_copy = cropped.copy()
	#print type(image1_copy)
	#contours,hierarchy = cv2.findContours(cropped,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	print ' Hi after contour finding function call'
	
	for i in range(len(sortContoursList)):
		[x,y,w,h] = sortContoursList[i]

		if  h>10:
			print 'hi inside h>50 if block'
			cv2.rectangle(image1_copy,(x,y),(x+w,y+h),(0,0,255),2)
			roi = image1_copy[y:y+h,x:x+w]    # roi = single character contour
			print type(roi),roi.shape
			roismall = cv2.resize(roi,(50,50))  # roismall = 50 x 50 character block
			print type(roismall),roismall.shape
			
			##########   Introduce HOG Descriptors here ###########
			deskewed_image = deskew(roismall)
			hogdata = hog(deskewed_image)
			print type(hogdata), hogdata
			
			
			#roismall = roismall.reshape((1,2500))
			#print 'ROISMALL shape : ' , roismall
			#roismall = np.float32(roismall)
			
			#hogdata = hogdata.reshape(-1,64)
			#hogdata = np.float32(hogdata)
			hogdata = np.float32(hogdata).reshape(-1,64)
			print 'Before SVM Prediction'
			results = svm.predict_all(hogdata)
			print 'After SVM Prediction'
			
			#retval, results, neigh_resp, dists = model.find_nearest(hogdata, k = 1)
			val = int(results[0][0])
			string = str(int((results[0][0])))
			#cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
			print 'Output String is : ', string ,  val
			cv2.putText(image1_copy,string,(x,y+h),0,1,(0,255,0))
			#cv2.imshow('Image',image1)
			
			OCRlist.append(val)
			#key = cv2.waitKey(0)

				

	cv2.imshow('Image',image1_copy)
	cv2.imwrite('OCR2.png',image1_copy)
	#cv2.imshow('out',out)
	cv2.waitKey(0)
	return OCRlist
	
	
	
def AsciiMapper(OCRAsciilist):
	OCRCharlist = []
	for i in OCRAsciilist:
		print chr(i)
		OCRCharlist.append(chr(i))
	OCRCharstring = ''.join(OCRCharlist)
	return OCRCharstring
	
	
def OCR(croplist,cropped,crop_color_string,cropped_color):
	
	#dilation_copy = dilation.copy()

	

	# img = cv2.imread(image_file)
	
	# print img.shape
	# img1 = cv2.imread(image_file,0) 					# Read the image as GRAYSCALE
	# cv.NamedWindow('Image')
	# cv2.resizeWindow('Image',100,100)
	# cv2.imshow('Original image',img1)
	# blur = cv2.GaussianBlur(img1,(5,5),0)	# Blur the image using GaussianBlur
	# ret3,th3 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	# changed this 0 to 100 ---- Create binary image using OTSU thresholding
	# imageinv =  255 - th3     				# Invert the image to have white text in black background
	# kernel = np.ones((5,5),np.uint8)
	# erosion = cv2.erode(imageinv,kernel,iterations = 1) 	# Erode the image
	# dilation = cv2.dilate(erosion,kernel,iterations = 2)	# Dilate the image
	# dilation_copy = dilation.copy()
	# contours,hierarchy = cv2.findContours(dilation_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	OCRlist = []
	#print OCRlist
	print 'Croplist is :',croplist

	print 'Inside OCR function'
	#image1 = tagPreprocessor(111,8,croplist[i])
	#img=cv.LoadImage(image_file)
	#out = np.zeros(image1.shape,np.uint8)
	#print type(image1)
	#image1 = np.asarray(image1[:,:])
	#print type(image1)
	image1_copy = cropped.copy()
	#print type(image1_copy)
	contours,hierarchy = cv2.findContours(cropped,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	print ' Hi after contour finding function call'
	
	for cnt in contours:
		if cv2.contourArea(cnt)>200:
			[x,y,w,h] = cv2.boundingRect(cnt)

			if  h>10:
				print 'hi inside h>50 if block'
				cv2.rectangle(cropped,(x,y),(x+w,y+h),(0,0,255),2)
				roi = cropped[y:y+h,x:x+w]
				print type(roi),roi.shape
				roismall = cv2.resize(roi,(50,50))
				print type(roismall),roismall.shape
				try:
					roismall = roismall.reshape((1,2500))
				except:
					print 'Hi raised exception'
				finally:
					roismall = np.float32(roismall)
					retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
					string = str(int((results[0][0])))
					#cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
					print 'Output string is : ',string
					cv2.putText(cropped_color,string,(x,y+h),0,1,(0,255,0))
					#cv2.imshow('Image',image1)
					
					OCRlist.append(string)
					#key = cv2.waitKey(0)

				

	cv2.imshow('Image',cropped_color)
	cv2.imwrite('OCR2.png',cropped_color)
	#cv2.imshow('out',out)
	cv2.waitKey(0)
	return OCRlist
	
	
def roundxy(pt):
    return (cv.Round(pt[0]), cv.Round(pt[1]))

	
def rotateImage1(skew_angle,box_vtx,image):
	#image_copy2 = image.copy()
	dilation_copy2 = dilation.copy()
	ori_image = cv2.imread(image)

	rows,cols,ch = ori_image.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),skew_angle,1)
	dst = cv2.warpAffine(dilation_copy2,M,(cols,rows))
	cv2.imwrite('Z_Rotated_image.jpg',dst)

	#b = cv.CreateMat(erosion.shape[0], erosion.shape[1], cv.CV_8UC1)
	#b = cv.fromarray(erosion)

	#cv.PolyLine(b, [box_vtx], 1, cv.CV_RGB(0, 255, 255), 1, cv.CV_AA)
	#cv2.polylines(erosion,[box_vtx],True,(255,0,0),2)
	cv2.polylines(dilation_copy2,np.array([box_vtx],dtype=np.int32),True,(255,0,0),2)

	#erosioncopy = erosion[530:603,992:1037]
	#image = np.asarray(b[:,:])
	#cv2.imshow('Boxed image',erosion)
	cv2.imwrite('Z_Boxed_image.jpg',dilation_copy2)
	return dst

		
def OCR2(croplist):
	OCRlist = []
	#print OCRlist
	for i in range(len(croplist)):
		print len(croplist)
		print croplist[i]
		image1=cv.LoadImage(croplist[i])
		tesseract.SetCvImage(image1,api)
		OCR=api.GetUTF8Text()
		#conf=api.MeanTextConf()
		print OCR
		OCRlist.append(OCR)
	return OCRlist

def cleanOCR1(OCRlist):
	#if OCRlist[0]
	# if any letters found in OCRlist[0] mrn => then change it to its corresponding digits say, b to 6, q to 9, l to 1, so on
	# if any letters found in OCRlist[1] unk1 => then change it to its corresponding digits as above
	newOCRlist =[]


	# mrn - digits |  name - letters  |  dob - both  |  sex - letter  |  att - letters
	
	
	######   Numbers correction
	
	nOCRlist = OCRlist[:5]  # splitting the digits placeholders
	print 'Inside cleanOCR',OCRlist
	print nOCRlist
	for j in range(len(nOCRlist)):
		print j
		newstr = nOCRlist[j]
		newlist = list(newstr)
	
		#posq = (OCRlist[0]).find('q')
		posq = [m.start() for m in re.finditer('q',newstr)]
		# finds all occurrences of a single character
		#print posq
		posb = [m.start() for m in re.finditer('b',newstr)]
		posl = [m.start() for m in re.finditer('l',newstr)]
		poso = [m.start() for m in re.finditer('o',newstr)]
		posexcl = [m.start() for m in re.finditer('!',newstr)]
		#pospipe = [m.start() for m in re.finditer('|',newstr)]   # pipe is not working as it is matching with 1
	
		if(posq):
			for i in range(len(posq)):
				print 'Inside posq'
				print len(posq)
				print 'i value is ',i
				newlist[posq[i]] = '9'
				print newlist
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(posb):
			for i in range(len(posb)):
				print 'Inside posb'
				print len(posb)
				print 'i value is ',i
				newlist[posb[i]] = '6'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(posl):
			for i in range(len(posl)):
				print 'Inside posl'
				print len(posl)
				print 'i value is ',i
				newlist[posl[i]] = '1'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(poso):
			for i in range(len(poso)):
				print 'Inside poso'
				print len(poso)
				print 'i value is ',i
				newlist[poso[i]] = '0'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(posexcl):
			for i in range(len(posexcl)):
				print 'Inside posexcl'
				print len(posexcl)
				print 'i value is ',i
				newlist[posexcl[i]] = '1'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
	# 	if(pospipe):
	# 		for i in range(len(pospipe)):
	# 			print 'Inside pospipe'
	# 			print len(pospipe)
	# 			print 'i value is ',i
	# 			newlist[pospipe[i]] = '1'
	# 			newstr = ''.join(newlist)
	# 			print newstr
	# 			#newOCRlist.append(newstr)
		
		newOCRlist.append(newstr)
	


	print newOCRlist
	
	
	
	
	####   Alphabets correction
	
	aOCRlist = OCRlist[5:]  # splitting the letters placeholders
	print aOCRlist
	for j in range(len(aOCRlist)):
		print j
		newstr = aOCRlist[j]
		newlist = list(newstr)
	
		#posq = (OCRlist[0]).find('q')
		pos8 = [m.start() for m in re.finditer('8',newstr)] # 8 => B
		# finds all occurrences of a single character
		#print posq
		pos1 = [m.start() for m in re.finditer('1',newstr)] # 1 => I
		posl = [m.start() for m in re.finditer('l',newstr)] # l => I
		poso = [m.start() for m in re.finditer('o',newstr)] # o => O
		pos0 = [m.start() for m in re.finditer('0',newstr)] # 0 => O
		pos2 = [m.start() for m in re.finditer('2',newstr)] # 2 => Z
		pos5 = [m.start() for m in re.finditer('5',newstr)] # 5 => S
		poss = [m.start() for m in re.finditer('s',newstr)] # s => S

		#posexcl = [m.start() for m in re.finditer('!',newstr)]
		#pospipe = [m.start() for m in re.finditer('|',newstr)]   # pipe is not working as it is matching with 1
	
		if(pos8):
			for i in range(len(pos8)):
				print 'Inside pos8'
				print len(pos8)
				print 'i value is ',i
				newlist[pos8[i]] = 'B'
				print newlist
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(pos1):
			for i in range(len(pos1)):
				print 'Inside pos1'
				print len(pos1)
				print 'i value is ',i
				newlist[pos1[i]] = 'I'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(posl):
			for i in range(len(posl)):
				print 'Inside posl'
				print len(posl)
				print 'i value is ',i
				newlist[posl[i]] = 'I'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(poso):
			for i in range(len(poso)):
				print 'Inside poso'
				print len(poso)
				print 'i value is ',i
				newlist[poso[i]] = 'O'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(pos0):
			for i in range(len(pos0)):
				print 'Inside pos0'
				print len(pos0)
				print 'i value is ',i
				newlist[pos0[i]] = '0'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(pos2):
			for i in range(len(pos2)):
				print 'Inside pos2'
				print len(pos2)
				print 'i value is ',i
				newlist[pos2[i]] = 'Z'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(pos5):
			for i in range(len(pos5)):
				print 'Inside pos5'
				print len(pos5)
				print 'i value is ',i
				newlist[pos5[i]] = 'S'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(poss):
			for i in range(len(poss)):
				print 'Inside poss'
				print len(poss)
				print 'i value is ',i
				newlist[poss[i]] = 'S'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)	
	# 	if(pospipe):
	# 		for i in range(len(pospipe)):
	# 			print 'Inside pospipe'
	# 			print len(pospipe)
	# 			print 'i value is ',i
	# 			newlist[pospipe[i]] = '1'
	# 			newstr = ''.join(newlist)
	# 			print newstr
	# 			#newOCRlist.append(newstr)
		
		newOCRlist.append(newstr)
	
	
	
	print newOCRlist
	return newOCRlist
	
	
def cleanOCR(OCRCharlist):
	#if OCRlist[0]
	# if any letters found in OCRlist[0] mrn => then change it to its corresponding digits say, b to 6, q to 9, l to 1, so on
	# if any letters found in OCRlist[1] unk1 => then change it to its corresponding digits as above
	newOCRlist =[]


	# mrn - digits |  name - letters  |  dob - both  |  sex - letter  |  att - letters
	
	
	######   Numbers correction
	
	nOCRlist = OCRlist[:4]  # splitting the digits placeholders
	print 'Inside cleanOCR',OCRlist
	print nOCRlist
	for j in range(len(nOCRlist)):
		print j
		newstr = nOCRlist[j]
		newlist = list(newstr)
	
		#posq = (OCRlist[0]).find('q')
		posq = [m.start() for m in re.finditer('q',newstr)]
		# finds all occurrences of a single character
		#print posq
		posb = [m.start() for m in re.finditer('b',newstr)]
		posl = [m.start() for m in re.finditer('l',newstr)]
		poso = [m.start() for m in re.finditer('o',newstr)]
		posexcl = [m.start() for m in re.finditer('!',newstr)]
		#pospipe = [m.start() for m in re.finditer('|',newstr)]   # pipe is not working as it is matching with 1
	
		if(posq):
			for i in range(len(posq)):
				print 'Inside posq'
				print len(posq)
				print 'i value is ',i
				newlist[posq[i]] = '9'
				print newlist
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(posb):
			for i in range(len(posb)):
				print 'Inside posb'
				print len(posb)
				print 'i value is ',i
				newlist[posb[i]] = '6'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(posl):
			for i in range(len(posl)):
				print 'Inside posl'
				print len(posl)
				print 'i value is ',i
				newlist[posl[i]] = '1'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(poso):
			for i in range(len(poso)):
				print 'Inside poso'
				print len(poso)
				print 'i value is ',i
				newlist[poso[i]] = '0'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(posexcl):
			for i in range(len(posexcl)):
				print 'Inside posexcl'
				print len(posexcl)
				print 'i value is ',i
				newlist[posexcl[i]] = '1'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
	# 	if(pospipe):
	# 		for i in range(len(pospipe)):
	# 			print 'Inside pospipe'
	# 			print len(pospipe)
	# 			print 'i value is ',i
	# 			newlist[pospipe[i]] = '1'
	# 			newstr = ''.join(newlist)
	# 			print newstr
	# 			#newOCRlist.append(newstr)
		
		newOCRlist.append(newstr)
	


	print newOCRlist
	
	
	
	
	####   Alphabets correction
	
	aOCRlist = OCRlist[5:]  # splitting the letters placeholders
	print aOCRlist
	for j in range(len(aOCRlist)):
		print j
		newstr = aOCRlist[j]
		newlist = list(newstr)
	
		#posq = (OCRlist[0]).find('q')
		pos8 = [m.start() for m in re.finditer('8',newstr)] # 8 => B
		# finds all occurrences of a single character
		#print posq
		
		pos1 = [m.start() for m in re.finditer('1',newstr)] # 1 => I
		posl = [m.start() for m in re.finditer('l',newstr)] # l => I
		poso = [m.start() for m in re.finditer('o',newstr)] # o => O
		pos0 = [m.start() for m in re.finditer('0',newstr)] # 0 => O
		pos2 = [m.start() for m in re.finditer('2',newstr)] # 2 => Z
		pos5 = [m.start() for m in re.finditer('5',newstr)] # 5 => S
		poss = [m.start() for m in re.finditer('s',newstr)] # s => S

		#posexcl = [m.start() for m in re.finditer('!',newstr)]
		#pospipe = [m.start() for m in re.finditer('|',newstr)]   # pipe is not working as it is matching with 1
	
		if(pos8):
			for i in range(len(pos8)):
				print 'Inside pos8'
				print len(pos8)
				print 'i value is ',i
				newlist[pos8[i]] = 'B'
				print newlist
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(pos1):
			for i in range(len(pos1)):
				print 'Inside pos1'
				print len(pos1)
				print 'i value is ',i
				newlist[pos1[i]] = 'I'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(posl):
			for i in range(len(posl)):
				print 'Inside posl'
				print len(posl)
				print 'i value is ',i
				newlist[posl[i]] = 'I'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(poso):
			for i in range(len(poso)):
				print 'Inside poso'
				print len(poso)
				print 'i value is ',i
				newlist[poso[i]] = 'O'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(pos0):
			for i in range(len(pos0)):
				print 'Inside pos0'
				print len(pos0)
				print 'i value is ',i
				newlist[pos0[i]] = '0'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(pos2):
			for i in range(len(pos2)):
				print 'Inside pos2'
				print len(pos2)
				print 'i value is ',i
				newlist[pos2[i]] = 'Z'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(pos5):
			for i in range(len(pos5)):
				print 'Inside pos5'
				print len(pos5)
				print 'i value is ',i
				newlist[pos5[i]] = 'S'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)
		if(poss):
			for i in range(len(poss)):
				print 'Inside poss'
				print len(poss)
				print 'i value is ',i
				newlist[poss[i]] = 'S'
				newstr = ''.join(newlist)
				print newstr
				#newOCRlist.append(newstr)	
	# 	if(pospipe):
	# 		for i in range(len(pospipe)):
	# 			print 'Inside pospipe'
	# 			print len(pospipe)
	# 			print 'i value is ',i
	# 			newlist[pospipe[i]] = '1'
	# 			newstr = ''.join(newlist)
	# 			print newstr
	# 			#newOCRlist.append(newstr)
		
		newOCRlist.append(newstr)
	
	
	
	print newOCRlist
	return newOCRlist
	
	
def JSONBuilder(newOCRlist, TagTemplates,tagIndex):
	nlist = []
	for i in range(TagTemplates[tagIndex][1]):
		nlist.append(TagTemplates[tagIndex][i+2][0])
	
	print nlist


	lst = []
	for i in range(len(newOCRlist)):	
		d = {}
		d[nlist[i]] = newOCRlist[i]
		lst.append(d)
		json_string = json.dumps(lst)
		#print json_string
	return json_string


def tagRecognizer(croplist):
	OCRlist = []
	#print OCRlist
	for i in range(len(croplist)):
		print len(croplist)
		print croplist[i]
		image1=cv.LoadImage(croplist[i])
		tesseract.SetCvImage(image1,api)
		OCR=api.GetUTF8Text()
		#conf=api.MeanTextConf()
		print OCR
		OCRlist.append(OCR)
	return OCRlist
	
				   
TagTemplates = [
					[
						'111',
						8,
						['T_mrn',470,1500,1,195],
						['T_unk1',1800,2500,1,190],
						['T_unk2',330,1400,490,660],
						['T_unk3',1800,2500,490,660],
						['T_dob',1,1100,330,500],
						['T_name',1,2000,190,330],						
						['T_sex',1600,1800,330,510],
						['T_att',415,2000,645,800],
					],
					[
						'222',
						9,
						['T_name',1,2000,1,100],
						['T_dob',1,1000,100,250],
						['T_sex',1000,1300,120,250],
						['T_mrn1',1300,2500,120,250],
						['T_MBNo',1,1750,230,380],
						['T_dept',1650,2500,250,430],
						['T_att',1,2000,400,500],
						['T_mrn2',800,2500,520,700],
						['T_lab',50,1500,700,800],
					]
					
			   ]
#tid = '111'
imgid = 111111
#image = '115_cut.jpg'
#image = 'output2.jpg'
#image = '1.png'
bin_n = 16
SZ = 50
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )


def mainProcess(tid,image):


	# img = cv2.imread('116_cut.jpg')
# 
# 	BLUE_MIN = np.array([90, 25, 35],np.uint8)
# 	BLUE_MAX = np.array([150, 255, 255],np.uint8)
# 	
# 	hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# 	cv2.imwrite('outputhsv.jpg',hsv_img)
# 	
# 	frame_threshed = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)
# 	cv2.imwrite('output2.jpg', frame_threshed)

	dilation = tagPreprocessor_HSV(tid,image)   # STEP 1 : calling the preprocessor for full dilated input image
	
	
	skew_angle,box_vtx = computeSkewAngle(dilation)   # STEP 2 : computing the skew angle of text
	
	#cropppped, crop_img_string = imgCropper(xmin,xmax,ymin,ymax,erosion,'cropppped')
	
	rot_image, rot_color_image, rot_img_string = rotateImage(skew_angle,box_vtx,dilation)  # STEP 3 : rotate the image with skew angle
	#print type(rot_image)
	#cv2.imwrite('Z_Rot_image_outside.jpg',rot_image)
	xmin,xmax,ymin,ymax = findExtremePoints(rot_image,rot_img_string) # STEP 4 : find the extreme points of aligned image
	try:
		crop_img_string,crop_image,crop_color_image = imgCropper1(xmin,xmax,ymin,ymax,rot_image,rot_color_image)  # STEP 5 : crop the aligned image
	except ValueError:
		print ' Exception raised as ValueError'
	finally:
		print crop_img_string
		resized_image,resized_color_image,resized_color_image_string = imgResizer(crop_img_string, 2500, 800,crop_color_image)   # STEP 6 : Resize the image to a specified template size
 		print 'After imgResizer function',type(resized_color_image)
		
		######### Next phase #########
		#resized_color_image = cv2.imread(
		resized_dilation = tagPreprocessor_HistEqualize(resized_color_image_string)
		
		OCR_train()  # STEP 7 : Train the classifier using KNN classifier with samples and responses got from Trainer and BootstrapTrainer
			
		
		samples = np.loadtxt('HOGfeatures10.data',np.float32)
		responses = np.loadtxt('HOGresp10.data',np.float32)
		responses = responses.reshape((responses.size,1))
		# model = cv2.KNearest()
	# 	model.train(samples,responses)
		
		
		svm = cv2.SVM()
		svm.train(samples,responses, params=svm_params)
		svm.save('svm_data.dat')
		
		OCRCharlist,tagIndex = Cropper(TagTemplates,tid)# STEP 8 : Crop every placeholder with cropping parameters from TagTemplates list
		print 'End of croplist',OCRCharlist

		json_string = JSONBuilder(OCRCharlist, TagTemplates,tagIndex)   # STEP 13 : Build a JSON object
		print 'JSON object is :     '
		print json_string
	return json_string
	
	'''
		rot_image, rot_img_string = rotateImage(skew_angle,box_vtx,cropppped)  # STEP 3 : rotate the image with skew angle
		print type(rot_image)
		cv2.imwrite('Z_Rot_image_outside1.jpg',rot_image)
		xmin,xmax,ymin,ymax = findExtremePoints(rot_image,rot_img_string) # STEP 4 : find the extreme points of aligned image
		crop_img_string,cropppped,crop_color_img = imgCropper1(xmin,xmax,ymin,ymax,rot_image)  # STEP 5 : crop the aligned image
		resized,resized_color = imgResizer(crop_img_string, 2500, 800,crop_color_img)   # STEP 6 : Resize the image to a specified template size
 		print 'After imgResizer function',type(resized)
 		
# 		#############
 		# To be added here Second time pipeline preprocessing + skew angle + rotateimage + findextremepoints + imgcropper1 + imgResizer	
# 		#############
# 		
 		image = 'Z_Resized_color_image.png'
 		dilation = tagPreprocessor2(tid,imgid,image)   # STEP 1 : calling the preprocessor for full dilated input image
	
	
		skew_angle,box_vtx = computeSkewAngle2(dilation)   # STEP 2 : computing the skew angle of text
		
		#cropppped, crop_img_string = imgCropper(xmin,xmax,ymin,ymax,erosion,'cropppped')
		
		rot_image, rot_img_string = rotateImage2(skew_angle,box_vtx,dilation)  # STEP 3 : rotate the image with skew angle
		print type(rot_image)
		cv2.imwrite('Z_Rot_image_outside2.jpg',rot_image)
		xmin,xmax,ymin,ymax = findExtremePoints(rot_image,rot_img_string) # STEP 4 : find the extreme points of aligned image
		try:
			crop_img_string,cropppped,crop_color_img = imgCropper2(xmin,xmax,ymin,ymax,rot_image)  # STEP 5 : crop the aligned image
		except ValueError:
			print ' Exception raised as ValueError'
		finally:
			print crop_img_string
			image = 'Z_Resized_color_image2.png'
			dilation = tagPreprocessor2(tid,imgid,image)   # STEP 1 : calling the preprocessor for full dilated input image

			skew_angle,box_vtx = computeSkewAngle3(dilation)
			rot_image, rot_img_string = rotateImage3(skew_angle,box_vtx,dilation)  # STEP 3 : rotate the image with skew angle
			print type(rot_image)
			cv2.imwrite('Z_Rot_image_outside3.jpg',rot_image)
			xmin,xmax,ymin,ymax = findExtremePoints(rot_image,rot_img_string) # STEP 4 : find the extreme points of aligned image
			crop_img_string,cropppped,crop_color_img = imgCropper3(xmin,xmax,ymin,ymax,rot_image)
			resized,resized_color = imgResizer3(crop_img_string, 2500, 800,crop_color_img)   # STEP 6 : Resize the image to a specified template size
			print 'After imgResizer function',type(resized)
			
			
			skew_angle,box_vtx = computeSkewAngle4(rot_image)
			rot_image, rot_img_string = rotateImage4(skew_angle,box_vtx,rot_image)  # STEP 3 : rotate the image with skew angle
			print type(rot_image)
			cv2.imwrite('Z_Rot_image_outside4.jpg',rot_image)
			xmin,xmax,ymin,ymax = findExtremePoints(rot_image,rot_img_string) # STEP 4 : find the extreme points of aligned image
			crop_img_string,cropppped,crop_color_img = imgCropper4(xmin,xmax,ymin,ymax,rot_image)
			resized,resized_color = imgResizer4(crop_img_string, 2500, 800,crop_color_img)   # STEP 6 : Resize the image to a specified template size
			print 'After imgResizer function',type(resized)
			
			
			OCR_train()  # STEP 7 : Train the classifier using KNN classifier with samples and responses got from Trainer and BootstrapTrainer
			
			
			samples = np.loadtxt('HOGsamples3.data',np.float32)
			responses = np.loadtxt('HOGresponses3.data',np.float32)
			responses = responses.reshape((responses.size,1))
			# model = cv2.KNearest()
		# 	model.train(samples,responses)
			
			
			svm = cv2.SVM()
			svm.train(samples,responses, params=svm_params)
			svm.save('svm_data.dat')
			
			
			
			OCRCharlist = Cropper(TagTemplates)# STEP 8 : Crop every placeholder with cropping parameters from TagTemplates list
			print 'End of croplist',OCRCharlist
			
			# convert the following Tesseract API call to our classifier's OCR API call
			# api = tesseract.TessBaseAPI()
			# api.Init(".","eng",tesseract.OEM_DEFAULT)
			# api.SetPageSegMode(tesseract.PSM_AUTO)
			
			
			#newOCRlist = cleanOCR(OCRCharlist)   # STEP 12 : clean OCR output
			#print newOCRlist
		
		
			json_string = JSONBuilder(OCRCharlist, TagTemplates)   # STEP 13 : Build a JSON object
			print 'JSON object is :     '
			print json_string
		# 	
			
			# print 'HIIII'
		# 
		# 	#OCRlist = tagRecognizer(croplist)    
		# 	
		# 	OCRlist = OCR(croplist)         # STEP 9 : call OCR(croplist[i]) for recognition of every placeholder
		# 
		# 	print OCRlist
		# 
		# 	newOCRlist = cleanOCR(OCRlist)   # STEP 10 : clean OCR output
		# 	print newOCRlist
		# 
		# 
		# 	json_string = JSONBuilder(newOCRlist, TagTemplates)   # STEP 11 : Build a JSON object
		# 	print 'JSON object is :     '
		# 	print json_string
		
			print ' W   O   W '
'''		