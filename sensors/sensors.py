import smbus

import imu as imu

bus = smbus.SMBus(1)#i2c_raspberry_pi_bus_number())
imu_controller = imu.IMU(bus, 0x68, 0x53, 0x1e, "IMU")

imu_controller.set_compass_offsets(9, -10, -140)


if __name__ == "__main__":
    while True:
        (pitch, roll, yaw) = imu_controller.read_pitch_roll_yaw()
        result = "%.2f %.2f %.2f" % (pitch, roll, yaw)
        print result
