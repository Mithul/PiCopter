#solenero.tech@gmail.com
#solenerotech.wordpress.com

#solenerotech 2013.09.06

from motor import Motor as motor
import time

mymotor1 = motor('m1', 17, simulation=False)    #RL
mymotor2 = motor('m1', 18, simulation=False)    #RR
mymotor3 = motor('m1', 27, simulation=False)    #FR
mymotor4 = motor('m1', 22, simulation=False)    #FL
#where 17 is  GPIO17 = pin 11

print('***Disconnect ESC power')
print('***then press ENTER')
res = raw_input()
mymotor1.start()
mymotor1.setW(100)
mymotor2.start()
mymotor2.setW(100)
mymotor3.start()
mymotor3.setW(100)
mymotor4.start()
mymotor4.setW(100)

#NOTE:the angular motor speed W can vary from 0 (min) to 100 (max)
#the scaling to pwm is done inside motor class
print('***Connect ESC Power')
print('***Wait beep-beep')

print('***then press ENTER')
res = raw_input()
mymotor1.setW(0)
mymotor2.setW(0)
mymotor3.setW(0)
mymotor4.setW(0)
print('***Wait N beep for battery cell')
print('***Wait beeeeeep for ready')
print('***then press ENTER')
res = raw_input()
print ('increase > a | decrease > z | save Wh > n | set Wh > h|quit > 9')

cycling = True
try:
    while cycling:
        res = raw_input()
        if res == 'a':
            mymotor1.increaseW(10)
            time.sleep(2)
            mymotor2.increaseW(10)
            time.sleep(2)
            mymotor3.increaseW(10)
            time.sleep(2)
            mymotor4.increaseW(10)
            time.sleep(2)
        if res == 'z':
            mymotor1.decreaseW(10)
            mymotor2.decreaseW(10)
            mymotor3.decreaseW(10)
            mymotor4.decreaseW(10)
        if res == 'n':
            mymotor1.saveWh()
            mymotor2.saveWh()
            mymotor3.saveWh()
            mymotor4.saveWh()
        if res == 'h':
            mymotor1.setWh()
            mymotor2.setWh()
            mymotor3.setWh()
            mymotor4.setWh()
        if res == '9':
            cycling = False
finally:
    # shut down cleanly
    mymotor1.stop()
    mymotor2.stop()
    mymotor3.stop()
    mymotor4.stop()
    print ("well done!")


