from array import *
import math

class Butterworth:
    def __init__(self, sampling, cutoff, order):

        self.n = int(round(order / 2))
        self.A = array("f", [])
        self.d1 = array("f", [])
        self.d2 = array("f", [])
        self.w0 = array("f", [])
        self.w1 = array("f", [])
        self.w2 = array("f", [])

        for ii in range(0, self.n):
            self.A.append(0.0)
            self.d1.append(0.0)
            self.d2.append(0.0)
            self.w0.append(0.0)
            self.w1.append(0.0)
            self.w2.append(0.0)


        a = math.tan(math.pi * cutoff / sampling)
        a2 = math.pow(a, 2.0)

        for ii in range(0, self.n):
            r = math.sin(math.pi * (2.0 * ii + 1.0) / (4.0 * self.n))
            s = a2 + 2.0 * a * r + 1.0
            self.A[ii] = a2 / s
            self.d1[ii] = 2.0 * (1 - a2) / s
            self.d2[ii] = -(a2 - 2.0 * a * r + 1.0) / s

    def filter(self, input):
        for ii in range(0, self.n):
            self.w0[ii] = self.d1[ii] * self.w1[ii] + self.d2[ii] * self.w2[ii] + input
            output = self.A[ii] * (self.w0[ii] + 2.0 * self.w1[ii] + self.w2[ii])
            self.w2[ii] = self.w1[ii]
            self.w1[ii] = self.w0[ii]

        return output