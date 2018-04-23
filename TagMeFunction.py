import cv2
import base64

def TagMeOCR(image):
	print 'Inside Image_to_String function', type(image)
	g = open("outinsideTagMeOCR.jpg", "w")
	g.write(base64.decodestring(image))
	g.close()
	b = cv2.imread("outinsideTagMeOCR.jpg")
	cv2.imshow('Inside TagMeOCR',b)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()
if __name__ == "__main__":

	print 'Outside Image_to_String function'	
	print 'Finally !!!!'
