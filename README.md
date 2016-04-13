# PiCopter
Code for the Quadcopter Project using Raspberry Pi done as a final year project at College of Engineering, Guindy.

## Requirements
  1. Require OpenCV 3.1.0 and OpenSfm installed for image processing
  2. Require pigpio library for motor control
  
## Usage 
### Start the server
 ```sh
  sudo python server2.py
  ```
  This will start a server on port 12345 and start balancing with minimal speed.
### Start the client
  ```sh
  sudo python client.py
  ```
  This will start a client program to control the height and the PID values of the quadcopter.
  

