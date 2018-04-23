import tornado.httpserver
import tornado.ioloop
import tornado.web
import pprint
import cv2
#import Image
#from tesseract import image_to_string
#from TagRecognizer_Box_HOG_KNN_Double import *
from TagMeFunction import TagMeOCR
import StringIO
import base64
from converter import image_to_base64, base64_to_image
 
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('<html><body>Send us a file!<br/><form enctype="multipart/form-data" action="/" method="post">'
                   '<input type="file" name="the_file">'
                   '<input type="text" name="template_id">'
                   '<input type="submit" value="Submit">'
                   '</form></body></html>')
 
    def post(self):
        self.set_header("Content-Type", "text/plain")
        self.write("You sent a file with name " + self.request.files.items()[0][1][0]['filename'] )
    # make a "memory file" using StringIO, open with PIL and send to tesseract for OCR
    #self.write(image_to_string(Image.open(StringIO.StringIO(self.request.files.items()[0][1][0]['body']))))
    	print type(self.request.files.items()[0][1][0]['filename']),self.request.files.items()[0][1][0]['filename']
    	print type(self.request.files.items()[0][1][0]['body'])
    	print type(StringIO.StringIO(self.request.files.items()[0][1][0]['filename']))
    	print 'Answer      :           ',self.request.value('template_id')
    	#image_data = image_to_base64(self.request.files.items()[0][1][0]['filename'])
    	image_data = base64.encodestring(open(self.request.files.items()[0][1][0]['filename'],"rb").read())
    	print len(image_data), type(image_data)
    	image_ext = self.request.files.items()[0][1][0]['filename']
    	image_ext1 = image_ext.split()
    	print image_ext1
    	if image_ext not in ['jpg','png','bmp']:
    		print 'Not allowed formats'
    	
    	#image_data = self.request.files.items()[0][1][0]['body'].decode("base64")
    	#print type(image_data)
    	g = open("out1.jpg", "w")
    	g.write(base64.decodestring(image_data))
    	#g.write(image_data)
    	g.close()
    	a = cv2.imread('out1.jpg')
    	cv2.imshow('Image',a)
    	TagMeOCR(image_data)
    	#### base64 string to image ####
    	# imgdata = base64.b64decode(imgstring)
#     	filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
#     	with open(filename, 'wb') as f:
#     		f.write(imgdata)
		#a = cv2.imread(self.request.files.items()[0][1][0]['filename'])
    	#cv2.imshow('Image',a)
    	#image_to_string(a)
    	#self.write(image_to_string(cv2.imread(self.request.files.items()[0][1][0]['filename'])))
    	#self.write(image_to_string(cv2.imread(self.request.files.items()[0][1][0]['body'])))
		
application = tornado.web.Application([
    (r"/", MainHandler),
])

# settings = {
# 				debug : True
# }
 
if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.instance().start()