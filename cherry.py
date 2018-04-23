import cherrypy
import cv2
import base64
from cherrypy import expose
from converter import image_to_base64
from TagRecognizer_Box_HOG_KNN_Double_Cherry import mainProcess

class Converter:
	@expose
	def tagMe(self,tid,image):
		image = image_to_base64(image)
		print tid
		print 'Inside Image_to_String function', type(image)
		print 'Inside tagMe in cherrypy server'
		g = open("insideTagMeCherrypy.jpg", "w")
		g.write(base64.decodestring(image))
		g.close()
		b = cv2.imread("insideTagMeCherrypy.jpg")
		print type(b)
		#cv2.imshow('Inside TagMeCherrypy',b)
		print 'Inside tagMe in cherrypy server'
		json_string = mainProcess(tid,image)
		#cv2.waitKey(10000)
		#cv2.destroyAllWindows()
		return json_string
	
	@expose
	def index(self):
		return "Hello World!"

	@expose
	def fahr_to_celc(self, degrees):
		temp = (float(degrees) - 32) * 5 / 9
		return "%.01f" % temp
	
	@expose
	def celc_to_fahr(self, degrees):
		temp = float(degrees) * 9 / 5 + 32
		return "%.01f" % temp

	
		
cherrypy.quickstart(Converter())