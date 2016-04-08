import RPi.GPIO as GPIO
import time

class Motor:

    def __init__(self, name, pin, kv=1050, WMin=0, WMax=100, debug=True, simulation=True):
        self.pin = pin
        self.name = name
        self.simulation = simulation
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin,GPIO.OUT)
        self.driverF = GPIO.PWM(pin, 50)

    def start(self):
        pass

    def stop(self):
        self.driverF.stop()

    def forward(self,speed):
        if self.simulation:
            return
        if speed>100:
            speed=100
            # print speed,' Speed too high reduced'
        if speed<0:
            speed=0
            # print speed,' Speed too low, set to 0'
        self.driverF.start(speed)

    def setW(self,speed):
        if self.simulation:
            return
        print self.name," speed ",speed
        self.forward(speed)

    def __exit__(self, type, value, traceback):
        self.driverF.stop()
        GPIO.cleanup()

