#solenero.tech@gmail.com
#solenerotech.wordpress.com

#solenerotech 2013.09.06

from motor import motor

mymotor1 = motor('m1', 17, simulation=False)
mymotor2 = motor('m1', 18, simulation=False)
#where 17 is  GPIO17 = pin 11

print('***Disconnect ESC power')
print('***then press ENTER')
res = raw_input()
mymotor1.start()
mymotor2.start()
mymotor1.setW(100)
mymotor2.setW(100)

#NOTE:the angular motor speed W can vary from 0 (min) to 100 (max)
#the scaling to pwm is done inside motor class
print('***Connect ESC Power')
print('***Wait beep-beep')

print('***then press ENTER')
res = raw_input()
mymotor1.setW(0)
mymotor2.setW(0)
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
            mymotor2.increaseW(10)
        if res == 'z':
            mymotor1.decreaseW(10)
            mymotor2.decreaseW(10)
        if res == 'n':
            mymotor1.saveWh()
            mymotor2.saveWh()
        if res == 'h':
            mymotor1.setWh()
            mymotor2.setWh()
        if res == '9':
            cycling = False
finally:
    # shut down cleanly
    mymotor1.stop()
    mymotor2.stop()
    print ("well done!")


