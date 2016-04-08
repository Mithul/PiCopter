import smbus
import time
import imu as imu

bus = smbus.SMBus(1)#i2c_raspberry_pi_bus_number())
imu_controller = imu.IMU(bus, 0x69, 0x53, 0x1e, "IMU")

imu_controller.set_compass_offsets(9, -10, -140)

from motor import motor

mymotor = motor('m1', 17, simulation=False)
#where 17 is  GPIO17 = pin 11

print('***Disconnect ESC power')
print('***then press ENTER')
res = raw_input()
mymotor.start()
mymotor.setW(100)

#NOTE:the angular motor speed W can vary from 0 (min) to 100 (max)
#the scaling to pwm is done inside motor class
print('***Connect ESC Power')
print('***Wait beep-beep')

print('***then press ENTER')
res = raw_input()
mymotor.setW(0)
print('***Wait N beep for battery cell')
print('***Wait beeeeeep for ready')
print('***then press ENTER')
res = raw_input()

while True:
    (pitch, roll, yaw) = imu_controller.read_pitch_roll_yaw()
    result = "%.2f %.2f %.2f" % (pitch, roll, yaw)
    mymotor.setW(int(roll*100))
    print result
    time.sleep(0.020)
