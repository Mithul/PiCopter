import smbus
import time
import sensors.imu2 as imu
from threading import Thread
from utils.butterworth import Butterworth
import os
import argparse


parser = argparse.ArgumentParser(
    description='Raspberry Pi Quadcopter Program.',
    usage='%(prog)s [options]')

parser.add_argument('--config', type=str, default="")
parser.add_argument('--motor', type=str, default="pigpio")

args = parser.parse_args()
print(args)

if args.motor == "pigpio":
    from motor.motor_pigpio import Motor
elif args.motor == "pwm":
    from motor.motor_pwm import Motor  # lint:ok
elif args.motor == "servo":
    from motor.motor_servo import Motor  # lint:ok
elif args.motor == "servoblaster":
    from motor.motor_servoblaster import Motor  # lint:ok
else:
    print("""Invalid motor controller selection. Please choose pigpio,
    pwm, servo, servoblaster""")
    exit(0)


os.nice(-10)


class Quad(object):

    def __init__(
            self,
            m1,
            m2,
            m3,
            m4,
            imu,
            s1=None,
            s2=None,
            s3=None,
            s4=None):
        self.motor_bl = m1
        self.motor_br = m4
        self.motor_fr = m3
        self.motor_fl = m2
        self.sensor_l = s1
        self.sensor_f = s2
        self.sensor_r = s3
        self.sensor_b = s4
        self.imu = imu

        self.thread = None
        self.running = False

        self.height = 10

        self.kp_x = 0
        self.kp_y = 0
        self.kp_z = 0
        self.ki_x = 0
        self.ki_y = 0
        self.ki_z = 0
        self.kd_x = 0
        self.kd_y = 0
        self.kd_z = 0

        self.limit_px = 20
        self.limit_py = 20
        self.limit_ix = 10
        self.limit_iy = 10

        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0

        self.trim_x = 0
        self.trim_y = 0
        self.trim_z = 0
        #x^ top is front
        # y->
        # z.
        dri_frequency = 40
        samples_per_motion = 4
        self.bfx = Butterworth(dri_frequency / samples_per_motion, 0.05, 8)
        self.bfy = Butterworth(dri_frequency / samples_per_motion, 0.05, 8)
        self.bfz = Butterworth(dri_frequency / samples_per_motion, 0.05, 8)

        self.imu.set_compass_offsets(9, -10, -140)
        self.motor_fl.start()
        self.motor_bl.start()
        self.motor_fr.start()
        self.motor_br.start()

        self.motor_fl.setW(0)
        self.motor_bl.setW(0)
        self.motor_fr.setW(0)
        self.motor_br.setW(0)

    def motor_init(self):
        self.motor_fl.setW(100)
        self.motor_bl.setW(100)
        self.motor_fr.setW(100)
        self.motor_br.setW(100)

        time.sleep(2)

        self.motor_fl.setW(0)
        self.motor_bl.setW(0)
        self.motor_fr.setW(0)
        self.motor_br.setW(0)

        time.sleep(1)

    def stop(self):
        self.running = False
        self.motor_fl.setW(0)
        self.motor_bl.setW(0)
        self.motor_fr.setW(0)
        self.motor_br.setW(0)
        self.motor_fl.stop()
        self.motor_bl.stop()
        self.motor_fr.stop()
        self.motor_br.stop()

    def set_PID(self, p, i, d):
        self.kp_x = p
        self.ki_x = i
        self.kd_x = d
        self.kp_y = p
        self.ki_y = i
        self.kd_y = d

    def compute_PID_output(
            self,
            kp,
            ki,
            kd,
            angle,
            old_i,
            old_angle,
            limit_p=100,
            limit_i=100,
            log=False):
        p = kp * angle
        i = old_i + ki * angle
        d = kd * (angle - old_angle)
        if p > limit_p:
            p = limit_p
        if i > limit_i:
            i = limit_i
        if log:
            log.write('\t' + str(p) + '\t' + str(i) + '\t' + str(d) + '\t')
        return [p + i + d, i]

    def balancer(self):
        print("Started Balancing")
        self.thread = Thread(target=self.balance)
        self.running = True
        self.thread.start()
        # self.balance()

    def set_zero_angle(self):
        (pitch, roll, yaw, _) = self.imu.read_pitch_roll_yaw()
        self.offset_x = pitch
        self.offset_y = roll
        self.offset_z = yaw

    def set_height(self, amt=0):
        self.height = amt

    def inc_height(self, amt=1):
        self.height = self.height + amt

    def dec_height(self, amt=1):
        self.height = self.height - amt

    def set_trims(self, x, y, z, relative=True):
        if relative:
            self.trim_x = self.trim_x + x
            self.trim_y = self.trim_y + y
            self.trim_z = self.trim_z + z
        else:
            self.trim_x = x
            self.trim_y = y
            self.trim_z = z

    def balance(self):
        pitch = 0
        roll = 0
        yaw = 0
        i_x = 0
        i_y = 0
        i_z = 0
        log = open('logs/motor.log', 'w')
        old_pitch, old_roll, old_yaw = 0, 0, 0
        log.write(
            """iteration\theight\tx-ouput\ty-output\tpitch\troll\ttime-taken
            \tgx\tgy\tax\tay\n""")
        i = 0
        ii = 0
        ustart_time = time.time()
        samples = 0
        while self.running:
            # print("Nothing")
            start_time = time.time()
            (pitch, roll, yaw, imu_connected) = self.imu.read_pitch_roll_yaw()
            ex = self.bfx.filter(pitch)
            ey = self.bfy.filter(roll)
            ez = self.bfz.filter(yaw)
            # (pitch, roll, yaw)=(ex, ey, ez)
            (_, _, gx, gy, _, ax, ay, _) = self.imu.read_all()
            axis_output = {'x': 0, 'y': 0, 'z': 0}
            if ii % 4 == 0:
                if imu_connected:
                    log.write(str(i) + '\t')
                    [axis_output['x'],
                     i_x] = self.compute_PID_output(self.kp_x,
                                                    self.ki_x,
                                                    self.kd_x,
                                                    pitch - self.offset_x,
                                                    i_x,
                                                    old_pitch,
                                                    self.limit_px,
                                                    self.limit_ix,
                                                    log)
                    [axis_output['y'],
                     i_y] = self.compute_PID_output(self.kp_y,
                                                    self.ki_y,
                                                    self.kd_x,
                                                    roll - self.offset_y,
                                                    i_y,
                                                    old_roll,
                                                    self.limit_py,
                                                    self.limit_iy,
                                                    log)
                    [axis_output['z'], i_z] = self.compute_PID_output(
                        self.kp_z, self.ki_z, self.kd_x,
                        yaw - self.offset_z, i_z, old_yaw)
                    axis_output['x'] = axis_output['x'] - self.trim_x
                    axis_output['y'] = axis_output['y'] - self.trim_y
                    self.motor_bl.setW(
                        int(self.height + axis_output['x'] / 2
                         + axis_output['y'] / 2))
                    self.motor_br.setW(
                        int(self.height + axis_output['x'] / 2
                        - axis_output['y'] / 2))
                    self.motor_fl.setW(
                        int(self.height - axis_output['x'] / 2
                         + axis_output['y'] / 2))
                    self.motor_fr.setW(
                        int(self.height - axis_output['x'] / 2
                        - axis_output['y'] / 2))
                    log.write(str(self.height) +
                              '\t' +
                              str(axis_output['x'] /
                                  2) +
                              '\t' +
                              str(axis_output['y'] /
                                  2))
                    log.write('\t' + str(pitch) + '\t' + str(roll))
                    end_time = time.time()
                    if (end_time - ustart_time) > 1:
                        ustart_time = time.time()
                        print(('samples ', samples, ustart_time))
                        samples = 0
                    print((pitch, roll, end_time, ustart_time, samples))
                    log.write('\t' + str(end_time - start_time))
                    log.write(
                        '\t' +
                        str(gx) +
                        '\t' +
                        str(gy) +
                        '\t' +
                        str(ax) +
                        '\t' +
                        str(ay))
                    log.write('\t' + str(ex) + '\t' + str(ey) + '\t' + str(ez))
                    log.write("\n")
                    i = i + 1
                else:
                    print("Warning IMU disconnected")
                old_pitch, old_roll, old_yaw = pitch, roll, yaw
            samples = samples + 1
            ii = ii + 1


if __name__ == "__main__":
    bus = smbus.SMBus(1)  # i2c_raspberry_pi_bus_number())
    imu_controller = imu.IMU(bus, 0x68, 0x53, 0x1e, "IMU")

    mymotor1 = Motor('m1', 17, debug=False, simulation=False)  # RL
    mymotor2 = Motor('m2', 18, debug=False, simulation=False)  # RR
    mymotor3 = Motor('m3', 27, debug=False, simulation=False)  # FR
    mymotor4 = Motor('m4', 22, debug=False, simulation=False)  # FL

    mymotor1.setMaxSpeed(30)
    mymotor2.setMaxSpeed(30)
    mymotor3.setMaxSpeed(30)
    mymotor4.setMaxSpeed(30)

    quadcopter = Quad(mymotor1, mymotor2, mymotor3, mymotor4, imu_controller)

    print ("""init > i | balance > b | stop > s | PID > p | set_zero > r
    \n increase_height > a | decrease_height > z | Emergency_zero > x""")

    if len(args.config) > 0:
        print("Reading file")
        config_file = open(args.config, 'r')
        import json
        pid = json.load(config_file)
        quadcopter.set_PID(pid['p'], pid['i'], pid['d'])
        print("PID set from file")

    cycling = True
    try:
        while cycling:
            res = raw_input()
            if res == 'i':
                quadcopter.motor_init()
            if res == 'a':
                quadcopter.inc_height(5)
            if res == 'z':
                quadcopter.dec_height(5)
            if res == 'b':
                quadcopter.balancer()
            if res == 'r':
                quadcopter.set_zero_angle()
            if res == 'x':
                quadcopter.set_height()
                quadcopter.stop()
            if res == 'p':
                p = raw_input()
                i = raw_input()
                d = raw_input()
                quadcopter.set_PID(p, i, d)
            if res == 's':
                quadcopter.stop()
                cycling = False
    finally:
        # shut down cleanly
        mymotor1.stop()
        mymotor2.stop()
        mymotor3.stop()
        mymotor4.stop()
        print ("well done!")

    quadcopter.stop()
