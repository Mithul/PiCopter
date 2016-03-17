import socket               # Import socket module
import json

from quad import Quad
import smbus
#import time
import sensors.imu2 as imu
from motor.motor_pigpio import Motor as motor

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

while True:
    try:
        msg = c.recv(1024)
        data = json.loads(msg)
        if 'reset' in data:
            quadcopter.set_zero_angle()
            continue
        print data['P']
        p = int(data['P'])
        i = int(data['I'])
        d = int(data['D'])
        height = int(data['pitch'])
        trim_x = int(data['trim_x'])
        trim_y = int(data['trim_y'])
        quadcopter.set_height(height)
        quadcopter.set_trims(trim_x, trim_y, 0, False)
        quadcopter.set_PID(p, i, d)
    except Exception as e:
        print e
        mymotor1.setMinSpeed(0)
        mymotor2.setMinSpeed(0)
        mymotor3.setMinSpeed(0)
        mymotor4.setMinSpeed(0)
        quadcopter.stop()
        break

    # print msg

c.close()                # Close the connection
