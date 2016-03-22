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


mymotor1 = motor('m1', 17, debug=False, simulation=False)  # RL
mymotor2 = motor('m2', 18, debug=False, simulation=False)  # RR
mymotor3 = motor('m3', 27, debug=False, simulation=False)  # FR
mymotor4 = motor('m4', 22, debug=False, simulation=False)  # FL

mymotor1.setMaxSpeed(60)
mymotor2.setMaxSpeed(60)
mymotor3.setMaxSpeed(60)
mymotor4.setMaxSpeed(60)

mymotor1.setMinSpeed(6)
mymotor2.setMinSpeed(6)
mymotor3.setMinSpeed(6)
mymotor4.setMinSpeed(6)

quadcopter = Quad(mymotor1, mymotor2, mymotor3, mymotor4, imu_controller)
quadcopter.dec_height(5)
quadcopter.balancer()
print 'Setting zero'
quadcopter.set_zero_angle()
time.sleep(2)
print 'Zero set'
p=30
i=0
d=55

quadcopter.set_PID(p, i, d)
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
            p = int(data['P'])
            i = float(data['I'])
            d = int(data['D'])
            quadcopter.set_PID(p, i, d)
        if 'height' in data:
            height = int(data['height'])
            quadcopter.set_height(height)
        if 'trim_x' in data:
            trim_x = int(data['trim_x'])
            trim_y = int(data['trim_y'])
            quadcopter.set_trims(trim_x, trim_y, 0, False)
        if 'X' in data:
            x = -float(data['Y'])/300
            y = -float(data['X'])/300
            z = float(data['Z'])/300
            height = int(data['Height'])
            #x = float(-y) / 300
            #y = float(x) / 300
            #z = float(z) / 300
            height = (float(height) + 300) / 6
            quadcopter.set_parameters(x, y, z, height)
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
