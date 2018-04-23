import base64
import cv2

# jpgtxt = base64.encodestring(open("111_cut.jpg","rb").read())
# #print jpgtxt
# print type(jpgtxt)
# f = open("out.txt", "w")
# f.write(jpgtxt)
# f.close()
# 
# 
# 
# # ----
# newjpgtxt = open("out.txt","rb").read()
# 
# g = open("111_cut_decoded.jpg", "w")
# g.write(base64.decodestring(newjpgtxt))
# g.close()



def image_to_base64(image_filename):
	image_string = base64.encodestring(open(image_filename,"rb").read())
	f = open("out.txt", "w")
	f.write(image_string)
	f.close()
	return image_string
	
	
def base64_to_image(image_string):
	#newjpgtxt = open("out.txt","rb").read()
	g = open("out.jpg", "w")
	g.write(base64.decodestring(image_string))
	g.close()
	
	
image_string = image_to_base64('111_cut.jpg')
base64_to_image(image_string)