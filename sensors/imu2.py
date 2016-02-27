import time

from adxl345 import ADXL345
from itg3205 import itg3205 as ITG3205
from hmc5883l import HMC5883L

class IMU(object):
    
    K = 0.98
    K1 = 1 - K
    
    Q_angle =  0.1                                                                   
    Q_gyro =  0.03                                                                 
    R_angle =  0.01                                                                     
    x_bias = 0                                                                          
    y_bias = 0                                                                          
    XP_00 = 0
    XP_01 = 0
    XP_10 = 0
    XP_11 = 0                                          
    YP_00 = 0
    YP_01 = 0
    YP_10 = 0
    YP_11 = 0                                          
    KFangleX = 0.0                                                                      
    KFangleY = 0.0     

    def __init__(self, bus, gyro_address, accel_address, compass_address, name, gyro_scale=0, accel_scale=ADXL345.AFS_2g):
        self.bus = bus
        self.gyro_address = gyro_address 
        self.accel_address = accel_address
        self.name = name
        self.gyro_scale = gyro_scale 
        self.accel_scale = accel_scale
        self.accelerometer = ADXL345(bus, accel_address, name + "-accelerometer", accel_scale)
        self.gyroscope = ITG3205(bus, gyro_address)
        self.compass = HMC5883L(bus, compass_address, name + "-compass")

        self.last_time = time.time()
        self.time_diff = 0

        self.pitch = 0
        self.roll = 0
        # take a reading from the device to allow it to settle after config changes
        self.read_all()
        # now take another to act a starting value
        self.read_all()
        self.pitch = self.rotation_x
        self.roll = self.rotation_y

    def read_all(self):
        '''Return pitch and roll in radians and the scaled x, y & z values from the gyroscope and accelerometer'''
        self.gyroscope.read_raw_data()
        self.accelerometer.read_raw_data()
        
        self.gyro_scaled_x = self.gyroscope.read_scaled_gyro_x()
        self.gyro_scaled_y = self.gyroscope.read_scaled_gyro_y()
        self.gyro_scaled_z = self.gyroscope.read_scaled_gyro_z()
        
        self.accel_scaled_x = self.accelerometer.read_scaled_accel_x()
        self.accel_scaled_y = self.accelerometer.read_scaled_accel_y()
        self.accel_scaled_z = self.accelerometer.read_scaled_accel_z()
        
        self.rotation_x = self.accelerometer.read_x_rotation(self.accel_scaled_x, self.accel_scaled_y, self.accel_scaled_z)
        self.rotation_y = self.accelerometer.read_y_rotation(self.accel_scaled_x, self.accel_scaled_y, self.accel_scaled_z)
        
        now = time.time()
        self.time_diff = now - self.last_time
        self.last_time = now 
        (self.pitch, self.roll) = self.comp_filter(self.rotation_x, self.rotation_y)
        
        # return (self.pitch, self.roll, self.gyro_scaled_x, self.gyro_scaled_y, self.gyro_scaled_z, self.accel_scaled_x, self.accel_scaled_y, self.accel_scaled_z)
        return (self.pitch, self.roll, self.gyro_scaled_x, self.gyro_scaled_y, self.gyro_scaled_z, self.accel_scaled_x, self.accel_scaled_y, self.accel_scaled_z)
        
    def read_x_rotation(self, x, y, z):
        return self.rotation_x

    def read_y_rotation(self, x, y, z):
        return self.rotation_y

    def comp_filter(self, current_x, current_y):
        new_pitch = IMU.K * (self.pitch + self.gyro_scaled_x * self.time_diff) + (IMU.K1 * current_x)
        new_roll = IMU.K * (self.roll + self.gyro_scaled_y * self.time_diff) + (IMU.K1 * current_y)
        return (new_pitch, new_roll)


    def read_pitch_roll_yaw(self):
        '''
        Return pitch, roll and yaw in radians
        '''
        try:
            (raw_pitch, raw_roll, self.gyro_scaled_x, self.gyro_scaled_y, \
                self.gyro_scaled_z, self.accel_scaled_x, self.accel_scaled_y, \
                self.accel_scaled_z) = self.read_all()
        
            now = time.time()
            self.time_diff = now - self.last_time
            self.DT = now - self.last_time
            self.last_time = now 
        
            (self.pitch, self.roll) = self.comp_filter(raw_pitch, raw_roll)
            # self.roll = -self.kalmanFilterX(self.accel_scaled_x, self.gyro_scaled_x)
            # self.pitch = self.kalmanFilterY(self.accel_scaled_y, self.gyro_scaled_y)
            self.yaw = self.compass.read_compensated_bearing(self.pitch, self.roll)
        except:
            return (self.pitch, self.roll, self.yaw, False)
        return (self.pitch, self.roll, self.yaw, True)

    def set_compass_offsets(self,x_offset, y_offset, z_offset):
        self.compass.set_offsets(x_offset, y_offset, z_offset)

    def kalmanFilterY(self, accAngle, gyroRate):                                        
                                                                                     
        self.KFangleY += self.DT * (gyroRate - self.y_bias)                                         
                                                                                     
        self.YP_00 +=  - self.DT * (self.YP_10 + self.YP_01) + self.Q_angle * self.DT                              
        self.YP_01 +=  - self.DT * self.YP_11                                                       
        self.YP_10 +=  - self.DT * self.YP_11                                                       
        self.YP_11 +=  + self.Q_gyro * self.DT                                                      
                                                                                     
        y = accAngle - self.KFangleY                                                      
        self.S = self.YP_00 + self.R_angle                                                          
        self.K_0 = self.YP_00 / self.S                                                                 
        self.K_1 = self.YP_10 / self.S                                                                 
                                                                                        
        self.KFangleY +=  self.K_0 * y                                                            
        self.y_bias  +=  self.K_1 * y                                                             
        self.YP_00 -= self.K_0 * self.YP_00                                                            
        self.YP_01 -= self.K_0 * self.YP_01                                                            
        self.YP_10 -= self.K_1 * self.YP_00                                                            
        self.YP_11 -= self.K_1 * self.YP_01                                                            
                                                                                        
        return self.KFangleY 

    def kalmanFilterX(self, accAngle, gyroRate):                                        
                                                                                     
        self.KFangleX += self.DT * (gyroRate - self.x_bias)                                         
                                                                                     
        self.XP_00 +=  - self.DT * (self.XP_10 + self.XP_01) + self.Q_angle * self.DT                              
        self.XP_01 +=  - self.DT * self.XP_11                                                       
        self.XP_10 +=  - self.DT * self.XP_11                                                       
        self.XP_11 +=  + self.Q_gyro * self.DT                                                      
                                                                                     
        x = accAngle - self.KFangleX                                                      
        self.S = self.XP_00 + self.R_angle                                                          
        self.K_0 = self.XP_00 / self.S                                                                 
        self.K_1 = self.XP_10 / self.S                                                                 
                                                                                        
        self.KFangleX +=  self.K_0 * x                                                            
        self.x_bias  +=  self.K_1 * x                                                             
        self.XP_00 -= self.K_0 * self.XP_00                                                            
        self.XP_01 -= self.K_0 * self.XP_01                                                            
        self.XP_10 -= self.K_1 * self.XP_00                                                            
        self.XP_11 -= self.K_1 * self.XP_01                                                            
                                                                                        
        return self.KFangleX                                                                