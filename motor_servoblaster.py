import time
import os
from threading import Thread

class Motor:

    def __init__(self, name, pin, kv=1050, WMin=0, WMax=100, debug=True, simulation=True):
        self.pin = pin
        self.name = name
        self.simulation = simulation
        self.speed=0
        self.running = False

        if pin==17:
            self.servoblaster_num="1"
        elif pin==18:
            self.servoblaster_num="2"
        elif pin==27:
            self.servoblaster_num="3" 
        elif pin==22:
            self.servoblaster_num="4"
        else:
            print "Invalid"

        self.thread = Thread(target=self.forward)
        self.running = True
        self.thread.start()


    def start(self):
        pass

    def stop(self):
        self.running = False
        os.system("echo "+self.servoblaster_num+"=100 > /dev/servoblaster")
        # self.driverF.stop()

    def forward(self):
        if self.simulation:
            return
        while self.running:
            if self.speed>100:
                self.speed=100
                # print self.speed,' Speed too high reduced'
            if self.speed<0:
                self.speed=0
                # print self.speed,' Speed too low, set to 0'
            os.system("echo "+self.servoblaster_num+"="+str(self.speed+100)+" > /dev/servoblaster")
        # self.driverF.start(speed)

    def setW(self,speed):
        if self.simulation:
            return
        self.speed=speed
        # print self.name," self.speed ",self.self.speed
        # self.forward(speed)

    def __exit__(self, type, value, traceback):
        self.stop()
        GPIO.cleanup()

