from cv2 import *
import picamera
from threading import Thread
import time
# initialize the camera
class PiCam:
	def __init__(self,name,interval=5):
		self.camera = picamera.PiCamera()
		self.name = name
		self.interval = interval
		self.running = False
		self.thread = None

	def shoot(self):
		print "Started Shooting"
		if self.thread == None:
			self.thread = Thread(target=self.take_pic)
			self.running = True
			self.thread.start()
		print "Im back"


	def take_pic(self):
		while self.running:
			im_name = self.name+"/"+str(int(time.time()))+".jpg"
			print "image",im_name
			self.camera.capture(im_name)
			time.sleep(self.interval)

	def stop(self):
		print "Stopped"
		if self.thread:
			self.thread.join(0)
		self.running = False

	def change_interval(self,interval):
		self.interval = interval

	def view_images(self):
		for i in xrange(self.threshold):
			im_name = self.name+"/"+str(i)+".jpg"
			cv2.imshow("Image",im_name)
			cv2.waitKey(0)

# p = PiCam('img',10)
# p.take_pic()
