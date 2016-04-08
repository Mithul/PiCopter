from motor.motor_pigpio import Motor
import sensors.imu2 as imu
import time
import smbus

mymotor1 = Motor('m1', 17, debug=False, simulation=False)  # RL
mymotor2 = Motor('m2', 18, debug=False, simulation=False)  # RR
mymotor3 = Motor('m3', 27, debug=False, simulation=False)  # FR
mymotor4 = Motor('m4', 22, debug=False, simulation=False)  # FL

bus = smbus.SMBus(1)
imu = imu.IMU(bus, 0x68, 0x53, 0x1e, "IMU")

def measure(s):
	st = time.time()
	m=-9999
	mean=0.0
	i=0
	while (time.time() - st) < s:
		g = imu.read_all()
		g = g[2:5]
		i=i+1
		# print g	
		total = abs(g[0]-0.0036424263999180213)+abs(g[1]+0.0036424263999180213)+abs(g[2]+0.013355563466366077) #- 0.00121414213331
		# print total
		if i>1:
			mean = mean/(i-1) + total/i
		else:
			mean = total
		if total>m:
			m=total
	return m*100,mean*1000




mymotor1.setW(0)
mymotor2.setW(0)
mymotor3.setW(0)
mymotor4.setW(0)

MOTOR2 = True
MOTOR3 = False
MOTOR1 = True
MOTOR4 = False

# print 'nothing',measure(5)
speed = 40

time.sleep(2)

wait = 2
calc = 5

while MOTOR2:
	mymotor2.setW(speed)
	k=0
	while k<calc:
		print 'motor 2',measure(1)
		k=k+1
	mymotor2.setW(0)
	k=0
	while k<wait:
		print k
		time.sleep(1)
		k=k+1

# while MOTOR1:
	mymotor1.setW(speed)
	k=0
	while k<calc:
		print 'motor 1',measure(1)
		k=k+1
	mymotor1.setW(0)
	k=0
	while k<wait:
		print k
		time.sleep(1)
		k=k+1

# while MOTOR3:
	mymotor3.setW(speed)
	k=0
	while k<calc:
		print 'motor 3',measure(1)
		k=k+1
	mymotor3.setW(0)
	k=0
	while k<wait:
		print k
		time.sleep(1)
		k=k+1

# while MOTOR4:
	mymotor4.setW(speed)
	k=0
	while k<calc:
		print 'motor 4',measure(1)
		k=k+1
	mymotor4.setW(0)
	k=0
	while k<wait:
		print k
		time.sleep(1)
		k=k+1

mymotor1.setW(0)
mymotor2.setW(0)
mymotor3.setW(0)
mymotor4.setW(0)