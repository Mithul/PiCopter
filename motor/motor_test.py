#solenero.tech@gmail.com
#solenerotech.wordpress.com

#solenerotech 2013.09.06

from motor4 import Motor as motor

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
print ('increase > a | decrease > z | save Wh > n | set Wh > h|quit > 9')

cycling = True
try:
    while cycling:
        res = raw_input()
        if res == 'a':
            mymotor.increaseW(10)
        if res == 'z':
            mymotor.decreaseW(10)
        if res == 'n':
            mymotor.saveWh()
        if res == 'h':
            mymotor.setWh()
        if res == '9':
            cycling = False
finally:
    # shut down cleanly
    mymotor.stop()
    print ("well done!")


