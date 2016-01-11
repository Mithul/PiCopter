import RPi.GPIO as GPIO
from threading import Thread
import time

GPIO.setmode(GPIO.BCM)
TRIG = 23

ECHO = 24
print "Distance Measurement In Progress"


class hc_sr04:
    def __init__(self, trig, echo, frequency):
        self.TRIG = trig
        self.ECHO = echo
        self.freq = frequency
        self.distance = 0
        GPIO.setup(self.TRIG,GPIO.OUT)
        GPIO.setup(self.ECHO,GPIO.IN)
        GPIO.output(self.TRIG, False)
        self.thread = None
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self.continuous_detection)
        self.thread.start()
        pass

    def detect(self):
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)

        GPIO.output(self.TRIG, False)
        while GPIO.input(self.ECHO)==0:
            pulse_start = time.time()
        while GPIO.input(self.ECHO)==1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration*17150
        self.distance = round(distance, 2)

    def continuous_detection(self):
        while self.running:
            self.detect()
            time.sleep(self.freq)

    def end(self):
        self.running = False

    def set_freq(self, freq):
        pass

    def __del__(self):
#        self.thread.join
        GPIO.cleanup()
    def get_dist(self):
        return self.distance


s1 =  hc_sr04(23,24,0.5)
s1.start()
raw_input()
#while True:
x=s1.get_dist()
print x
raw_input()
s1.end()
