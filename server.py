import socket               # Import socket module
import json

from quad import Quad
import smbus
#import time
import sensors.imu2 as imu
from motor.motor_pigpio import Motor as motor

import time
import traceback

s = socket.socket()         # Create a socket object
host = '0.0.0.0'            # Get local machine name
port = 12345                # Reserve a port for your service.
s.bind((host, port))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
c, addr = s.accept()     # Establish connection with client.
print 'Got connection from', addr

bus = smbus.SMBus(1)  # i2c_raspberry_pi_bus_number())
imu_controller = imu.IMU(bus, 0x68, 0x53, 0x1e, "IMU")


mymotor1 = motor('br', 17, debug=False, simulation=False)  # RL
mymotor2 = motor('fr', 18, debug=False, simulation=False)  # RR
mymotor3 = motor('bl', 27, debug=False, simulation=False)  # FR
mymotor4 = motor('fl', 22, debug=False, simulation=False)  # FL

mymotor1.setMaxSpeed(80)
mymotor2.setMaxSpeed(80)
mymotor3.setMaxSpeed(80)
mymotor4.setMaxSpeed(80)

mymotor1.setMinSpeed(5)
mymotor2.setMinSpeed(5)
mymotor3.setMinSpeed(5)
mymotor4.setMinSpeed(5)

quadcopter = Quad(mymotor1, mymotor2, mymotor3, mymotor4, imu_controller)
quadcopter.dec_height(5)
print 'Setting zero'
quadcopter.load_zero()
print quadcopter.zero_x,quadcopter.zero_y,quadcopter.zero_z
time.sleep(2)
quadcopter.balancer()
print 'Zero set'
pr=0
p=0
i=0
d=0

# quadcopter.set_PID(p, i, d, pr)
quadcopter.set_trims(0, 7, 0, False)
success = 5
while True:
    try:
        msg = c.recv(1024)
        print 'message'+msg
        data = json.loads(msg)
        if 'reset' in data:
            quadcopter.set_zero_angle()
            continue
        if 'P' in data:
            print data['P']
            pr = int(data['PR'])
            p = int(data['P'])
            i = float(data['I'])
            d = int(data['D'])
            quadcopter.set_PID(p, i, d, pr)
        if 'height' in data:
            height = int(data['height'])
            quadcopter.set_height(height)
        if 'trim_x' in data:
            trim_x = int(data['trim_x'])
            trim_y = int(data['trim_y'])
            quadcopter.set_trims(trim_x, trim_y, 0, False)
        if 'X' in data:
            x = -float(data['Y'])/200
            y = -float(data['X'])/200
            z = 0
            print "message",x,y,z
            height = int(data['Height'])
            #x = float(-y) / 300
            #y = float(x) / 300
            #z = float(z) / 300
            height = (float(height) + 300) / 6
            quadcopter.set_parameters(x, y, z, height)
        if 'running' in data:
            if data['running']==False:
                raise
        if success<5:
            success = success + 1
        print success
    except ValueError as e:
        print e
        if success < 0:
            mymotor1.setMinSpeed(0)
            mymotor2.setMinSpeed(0)
            mymotor3.setMinSpeed(0)
            mymotor4.setMinSpeed(0)
            quadcopter.stop()
            break
            raise
        else:
            success = success - 1
    except Exception as e:
        print e
        mymotor1.setMinSpeed(0)
        mymotor2.setMinSpeed(0)
        mymotor3.setMinSpeed(0)
        mymotor4.setMinSpeed(0)
        quadcopter.stop()
        break

traceback.print_exc()
    # print msg

c.close()                # Close the connection
