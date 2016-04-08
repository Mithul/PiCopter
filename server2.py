import socket               # Import socket module
import json

from quad import Quad
import smbus
#import time
import sensors.imu2 as imu
from motor.motor_pigpio import Motor as motor

import time
import traceback

class Server(object):

	CONTROL_MESSAGE = 1
	CALIBRATE_MESSAGE = 2
	ACTION_MESSAGE = 3
	TUNE_MESSAGE = 4

	def __init__(self,quad, motor1, motor2, motor3, motor4):
		self.server_sock = socket.socket()         # Create a socket object
		host = '0.0.0.0'            # Get local machine name
		port = 12345                # Reserve a port for your service.
		self.server_sock.bind((host, port))        # Bind to the port

		self.server_sock.listen(5)                 # Now wait for client connection.

		self.client = None
		self.quadcopter = quad
		self.motor1 = motor1
		self.motor2 = motor2
		self.motor3 = motor3
		self.motor4 = motor4

		quadcopter.dec_height(5)
		print 'Setting zero'
		self.quadcopter.load_zero()
		self.quadcopter.set_zero_z()
		print self.quadcopter.zero_x,self.quadcopter.zero_y,self.quadcopter.zero_z
		time.sleep(2)
		self.quadcopter.balancer()
		print 'Zero set'
		pr=0
		p=0
		i=0
		d=0

		# quadcopter.set_PID(p, i, d, pr)
		self.quadcopter.set_trims(0, 7, 0, False)
		pass

	def connect(self):
		self.client, addr = self.server_sock.accept()     # Establish connection with client.
		print 'Got connection from', addr
		self.client.settimeout(0.9)
		# pass

	def start(self):
		success = 0
		height = 0
		last_recv = 0
		while True:
		    try:
		        msg = self.client.recv(1024)
		        print 'message'+msg+' '+str(time.time()-last_recv)
		        last_recv = time.time()
		        data = json.loads(msg)
		        # if 'reset' in data:
		        #     self.quadcopter.set_zero_angle()
		        #     continue
		        # if 'P' in data:
		        #     print data['P']
		        #     pr = int(data['PR'])
		        #     p = int(data['P'])
		        #     i = float(data['I'])
		        #     d = int(data['D'])
		        #     self.quadcopter.set_PID(p, i, d, pr)
		        # if 'height' in data:
		        #     height = int(data['height'])
		        #     self.quadcopter.set_height(height)
		        # if 'trim_x' in data:
		        #     trim_x = int(data['trim_x'])
		        #     trim_y = int(data['trim_y'])
		        #     self.quadcopter.set_trims(trim_x, trim_y, 0, False)
		        # if 'X' in data:
		        #     x = -float(data['Y'])/200
		        #     y = -float(data['X'])/200
		        #     z = 0
		        #     print "message",x,y,z
		        #     height = int(data['Height'])
		        #     #x = float(-y) / 300
		        #     #y = float(x) / 300
		        #     #z = float(z) / 300
		        #     height = (float(height) + 300) / 6
		        #     self.quadcopter.set_parameters(x, y, z, height)
		        if data['messageType']==Server.CONTROL_MESSAGE:
		        	x = -float(data['Y'])/100
		        	y = -float(data['X'])/100
		        	z = 0
		        	height = (int(data['Height'])+70)/2
		        	print "message",x,y,z,height
		        	self.quadcopter.set_parameters(x, y, z, height)
		        if data['messageType']==Server.TUNE_MESSAGE:
		     		print data['P']
		        	pr = int(data['PR'])
		        	p = int(data['P'])
		        	i = float(data['I'])
		        	d = int(data['D'])
		        	self.quadcopter.set_PID(p, i, d, pr)
		        if data['messageType']==Server.CALIBRATE_MESSAGE:
		     		trim_x = int(data['trim_x'])
		        	trim_y = int(data['trim_y'])
		        	self.quadcopter.set_trims(trim_x, trim_y, 0, False)
		        if 'running' in data:
		            if data['running']==False:
		                raise Exception
		        if success<5:
		            success = success + 1
		        print success
		    except ValueError as e:
		        print e
		        if success < 0:
		            self.quadcopter.stop()
		            self.motor1.setMinSpeed(0)
		            self.motor2.setMinSpeed(0)
		            self.motor3.setMinSpeed(0)
		            self.motor4.setMinSpeed(0)
		            break
		            raise
		        else:
		            success = success - 1
		    except Exception as e:
		        print e
		        print str(time.time()-last_recv)
		        self.quadcopter.stop()
		        self.motor1.setMinSpeed(0)
		        self.motor2.setMinSpeed(0)
		        self.motor3.setMinSpeed(0)
		        self.motor4.setMinSpeed(0)
		        break

		traceback.print_exc()

		pass

	def stop(self):
		c.close()                # Close the connection
		pass

if __name__ == "__main__":

	bus = smbus.SMBus(1)  # i2c_raspberry_pi_bus_number())
	imu_controller = imu.IMU(bus, 0x68, 0x53, 0x1e, "IMU")


	mymotor1 = motor('br', 17, debug=False, simulation=False)  # RL
	mymotor2 = motor('fr', 18, debug=False, simulation=False)  # RR
	mymotor3 = motor('bl', 27, debug=False, simulation=False)  # FR
	mymotor4 = motor('fl', 22, debug=False, simulation=False)  # FL

	quadcopter = Quad(mymotor1, mymotor2, mymotor3, mymotor4, imu_controller)

	server = Server(quadcopter, mymotor1, mymotor2, mymotor3, mymotor4)

	server.connect()
	server.start()
