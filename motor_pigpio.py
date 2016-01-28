import pigpio
import time

class Motor:

    def __init__(self, name, pin, kv=1050, WMin=0, WMax=100, debug=True, simulation=True):
        self.pin = pin
        self.name = name
        self.simulation = simulation
        self.speed=0
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(pin,GPIO.OUT)
        self.driver = pigpio.pi()

    def start(self):
        pass

    def stop(self):
        self.driver.set_servo_pulsewidth(self.pin, 0)

    def forward(self,speed):
        if self.simulation:
            return
        if speed>100:
            self.speed=100
            # print speed,' Speed too high reduced'
        if speed<0:
            self.speed=0
            # print speed,' Speed too low, set to 0'
        speed=speed*5
        self.driver.set_servo_pulsewidth(self.pin, 1000+speed)

    def increaseW(self,inc=1):
        if self.simulation:
            return
        self.speed=self.speed+inc
        self.forward(self.speed)
        print self.name," speed ",self.speed

    def decreaseW(self,inc=1):
        if self.simulation:
            return
        self.speed=self.speed-inc
        self.forward(self.speed)
        print self.name," speed ",self.speed

    def setW(self,speed):
        if self.simulation:
            return
        print self.name," speed ",speed
        self.forward(speed)

    def __exit__(self, type, value, traceback):
        self.driverF.stop()
        GPIO.cleanup()

