from cv2 import *
import picamera
from time import sleep
# initialize the camera
class PiCam:
	def __init__(self,name,threshold=5):
		self.camera = picamera.PiCamera()
		self.name = name
		self.threshold = threshold

	def take_pic(self):
		for i in xrange(self.threshold):
			im_name = self.name+"/"+str(i)+".jpg"
			self.camera.capture(im_name)
			sleep(2)

	def change_threshold(self,new_threshold):
		self.threshold = new_threshold

	def view_images(self):
		for i in xrange(self.threshold):
			im_name = self.name+"/"+str(i)+".jpg"
			cv2.imshow("Image",im_name)
			cv2.waitKey(0)
			