import cherrypy
import cv2
import base64
from cherrypy import expose
from converter import image_to_base64
from TagRecognizer_Box_HOG_KNN_Double_Cherry_files import mainProcess
from TagRecognizer_Box_HOG_KNN_Double_Cherry import mainProcess1

class Root(object):
    def __init__(self):
        self.app1 = App1()
        self.app2 = App2()
        self.app3 = Converter()
        self.app4 = FileDemo()

class App1(object):
    @expose
    def index(self):
        return "Hello world from app1"

class App2(object):
    @expose
    def index(self):
        return "Hello world from app2"

class FileDemo(object):

    def index(self):
        return """
        <html><body>
            <h2>Upload a file</h2>
            <form action="upload" method="post" enctype="multipart/form-data">
            filename: <input type="file" name="myFile" /><br />
            Template ID: <input type="text" name="tid" /><br />
            <input type="submit" />
            </form>
            <h2>Download a file</h2>
            <a href='download'>This one</a>
        </body></html>
        """
    index.exposed = True
    
    def upload(self, myFile,tid):
        out = """<html>
        <body>
            myFile length: %s<br />
            myFile filename: %s<br />
            myFile mime-type: %s<br />
            
           
        </body>
        </html>"""

        # Although this just counts the file length, it demonstrates
        # how to read large files in chunks instead of all at once.
        # CherryPy reads the uploaded file into a temporary file;
        # myFile.file.read reads from that.
        size = 0
        g = open("outcherryfilesOCR.jpg", "w")
        while True:
            data = myFile.file.read(8192)
            
            g.write(data)
            #print data
            if not data:
                break
            size += len(data)
        #image_string = base64.encodestring(data)
        #print type(data),data
        #g.write(data)
        g.close()
        
        a = cv2.imread('outcherryfilesOCR.jpg',0)
        print type(a)
        json_string = mainProcess(tid,a)
        
        return json_string
    upload.exposed = True

    def download(self):
        path = os.path.join(absDir, "pdf_file.pdf")
        return static.serve_file(path, "application/x-download",
                                 "attachment", os.path.basename(path))
    download.exposed = True


# import os.path
# tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')


class Converter(object):
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
		json_string = mainProcess1(tid,image)
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

if __name__ == '__main__':
    hostmap = {
        'company.com:8080': '/app1',
        'home.net:8080': '/app2',
        'tagmeold.com:8080': '/app3',
        'tagme.com:8080': '/app4',
    }

    config = {
        'request.dispatch': cherrypy.dispatch.VirtualHost(**hostmap)
    }

    cherrypy.quickstart(Root(), '/', {'/': config})
    
    
    

