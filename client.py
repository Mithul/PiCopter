#!/usr/bin/python           # This is client.py file

import socket               # Import socket module
import time
import json
from Tkinter import *

s = socket.socket()         # Create a socket object
host = 	'raspberrypi.local' # Get local machine name
port = 12345                # Reserve a port for your service.

s.connect((host, port))
run = False

def show_values():
	msg =  {'pitch': pitch.get(),'roll': roll.get(),'P': P.get(),'I': I.get(),'D': D.get(), 'run': run}
	msg1 = json.dumps(msg)
	print msg1
	s.send(msg1)

def stop():
	global run
	run = False

def reset():
	s.send(json.dumps({'reset':True}))

def start():
	global run
	run = True

master = Tk()
pitch = Scale(master, from_=0, to=100, tickinterval=10)
P = Scale(master, from_=0, to=500, tickinterval=100, label='P', orient=HORIZONTAL, length=600)
I = Scale(master, from_=0, to=500, tickinterval=100, label='I', orient=HORIZONTAL, length=600)
D = Scale(master, from_=0, to=500, tickinterval=100, label='D', orient=HORIZONTAL, length=600)
P.set(0)
I.set(0)
D.set(0)
P.pack()
I.pack()
D.pack()
pitch.set(5)
pitch.pack()
roll = Scale(master, from_=0, to=100, length=600,tickinterval=10, orient=HORIZONTAL) 
roll.set(5)
roll.pack()
Button(master, text='Show', command=show_values).pack()
Button(master, text='Stop', command=stop).pack()
Button(master, text='Reset', command=reset).pack()

mainloop()


s.close                     # Close the socket when done