from motor.motor_pigpio import Motor
mymotor1 = Motor('m1', 17, debug=False, simulation=False)  # RL
mymotor2 = Motor('m2', 18, debug=False, simulation=False)  # RR
mymotor3 = Motor('m3', 27, debug=False, simulation=False)  # FR
mymotor4 = Motor('m4', 22, debug=False, simulation=False)  # FL

mymotor1.setW(0)
mymotor2.setW(0)
mymotor3.setW(0)
mymotor4.setW(0)