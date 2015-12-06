class quad(object)
    def __init__(self, m1, m2, m3, m4, s1, s2, s3, s4, imu):
        self.motor_fl=m1
        self.motor_fr=m2
        self.morot_bl=m3
        self.motor_br=m4
        self.sensor_l=s1
        self.sensor_f=s2
        self.sensor_r=s3
        self.sensor_b=s4
        self.imu=imu
        self.kp_x=0
        self.kp_y=0
        self.kp_z=0
        self.ki_x=0
        self.ki_y=0
        self.ki_z=0
        self.kd_x=0
        self.kd_y=0
        self.kd_z=0
        self.height=0
        #x^ top is front
        #y->
        #z.


    def compute_PID_output(self, kp, ki, kd, angle, old_i, old_angle)
        p = kp*angle
        i = old_i+ki*angle
        d = kd*(angle-old_angle)
        return [p+i+d,i]

    def balance(self):
        while True:
            orientation = imu.read()
            axis_output = {x: 0, y: 0, z: 0}
            [axis_output['x'],i_x]=compute_PID_output(self.kp_x, self.ki_x, self.kd_x, i_x, orientation['x'])
            [axis_output['y'],i_y]=compute_PID_output(self.kp_y, self.ki_y, self.kd_x, i_y, orientation['y'])
            [axis_output['z'],i_z]=compute_PID_output(self.kp_z, self.ki_z, self.kd_x, i_z, orientation['z'])
            axis_output['x']=axis_output['x']-self.offset_x
            axis_output['y']=axis_output['y']-self.offset_y
            axis_output['z']=axis_output['z']-self.offset_z
            self.motor_fl.set_W(self.height+axis_output['x']/2+axis_output['y']/2)
            self.motor_fr.set_W(self.height+axis_output['x']/2-axis_output['y']/2)
            self.motor_bl.set_W(self.height-axis_output['x']/2+axis_output['y']/2)
            self.motor_br.set_W(self.height-axis_output['x']/2-axis_output['y']/2)
