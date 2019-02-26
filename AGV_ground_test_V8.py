'''
Date: 
Status: Working
Description:
'''


# Helper libraries
import requests
import sys
import numpy as np
import thread
import time
from time import sleep
import copy
import RPi.GPIO as GPIO
import lidarLiteV3
import requests
from PIL import Image
from StringIO import StringIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

#=================================================================================================
#IP camera rotation control functions.

user='admin'
passw='Camuaslab'
ip = "192.168.1.8"

right1 ='http://'+ ip +'/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=right'
left1 =  'http://'+ ip +'/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=left'
up1 =   'http://'+ ip +'/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=up'
down1 = 'http://'+ ip +'/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=down'
stop1 = 'http://'+ ip +'/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=stop'
home1 = 'http://'+ ip +'/web/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act=home'

def right():
      r = requests.get(right1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))
       
def left():
      r = requests.get(left1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))

def up():
      r = requests.get(up1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))
def down():
      r = requests.get(down1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))
def stop():
      s = requests.get(stop1,auth=(user,passw))
      s = requests.get(stop1,auth=(user,passw))

def rightrotate():
      r = requests.get(right1,auth=(user,passw))
  
def leftrotate():
     r = requests.get(left1,auth=(user,passw))
def uprotate():
     r = requests.get(up1,auth=(user,passw))
def downrotate():
     r = requests.get(down1,auth=(user,passw))
def home():
     r = requests.get(home1,auth=(user,passw))


#====================================================================================================

#Lidar setup
lidar= lidarLiteV3.Lidar_Lite()
connected =lidar.connect(1)
if connected <-1:
	print (connected)


# ======================================================================================================
# Lidar platform (pan) motion control module. In this program the lidar pan-tilt and distance collection is done by thread program. 
# Initially Keep the lidar sensor beam's direction parallel to the heading angle of the vehicle.

# Arduino control pins
forwardstate = True
stopstate = True
leftstate = True
rightstate = True
forwardpin = 31
botstop = 32
leftpin = 29
rightpin = 26
GPIO.setup(forwardpin, GPIO.OUT, initial = 1)
GPIO.setup(botstop, GPIO.OUT, initial = 1)
GPIO.setup(leftpin, GPIO.OUT, initial = 1)
GPIO.setup(rightpin, GPIO.OUT, initial = 1)


#====================================================================================================
#set the below parameters during changes in angles,safe distance etc.
obstacleDetected = False #set this variable if obstacle is detected.
leftAngle = -60;
rightAngle = 60;
normalizeAngle = 60.0;
threshold = 100 # threshold distance (1.2 meters), less than this is not considered safe
safeDistance = threshold + 50
totalDistance = 1001 # Maximum distance value from lidar. 10 meters. The values are taken in centimeter.
normalizeDistance = 1000.0
points = 121; # points per sweep (equal to number of angles)


sweepData = {}; # lidar sweep data is stored in this dictionary


# ======================================================================================================
# This module uses the saved weights and biases to provide guidance for the AGV. 

def callNeuralNetwork(): 
	r = requests.get('http://'+ ip +'/tmpfs/auto.jpg?',auth=(user,passw)) #provide IP address of the camera.
        i = Image.open(StringIO(r.content))
        i.save("FOVpicture.jpg")
        url = 'http://192.168.1.2:5000/' #provide IP address of the CNN server.
        files = {'file': open('/home/pi/Desktop/Hybrid_AGV(Gokul FYP)/FOVpicture.jpg', 'rb')} #The path of the picture taken by camera.
        result = requests.post(url, files=files)
        print "CNN response:"
        print result.text
        
	if result.text == "Forward":
                global forwardstate
                forwardstate = not forwardstate
                GPIO.output(forwardpin, forwardstate)
                
        elif result.text == "Left":
                global leftstate
                leftstate = not leftstate
                GPIO.output(leftpin,leftstate)
                
        elif guidance == "Right":
                global rightstate
                rightstate = not rightstate
                GPIO.output(rightpin,rightstate)
                
        else:
                global stopstate
                stopstate = not stopstate
                GPIO.output(botstop, stopstate)


# ======================================================================================================
# This is a thread method that collects distances when the servo motor is moving between the start angle and end angle.

def lidarGetDistance(startAngle,endAngle):
        global sweepData,threshold,stopstate,safeDistance,totalDistance;
        if startAngle < endAngle:
                for angle in range(startAngle,endAngle+1,1):
                        try:
                          sweepData[angle]=lidar.getDistance();
                          if(sweepData[angle] > totalDistance):
                                  sweepData[angle] = totalDistance;
                          if(sweepData[angle] < safeDistance and angle >=-20 and angle <=20):
                                stopstate = not stopstate;
                                GPIO.output(botstop, stopstate)
                                obstacleDetected = True
                        except IOError:
                          #print "\nInput/Output Exception occurred in thread...!\n";
                          sweepData[angle] = int(np.random.randint(0,totalDistance,1));
                        sleep(0.04);
        else:
                for angle in range(startAngle,endAngle-1,-1):
                        try:
                          sweepData[angle]=lidar.getDistance();
                          if(sweepData[angle] > totalDistance and angle >=-20 and angle <=20):
                                  sweepData[angle] = totalDistance;
                          if(sweepData[angle] < safeDistance):
                                stopstate = not stopstate;
                                GPIO.output(botstop, stopstate)
                                obstacleDetected = True
                        except IOError:
                          #print "\nInput/Output Exception occurred in thread...!\n";
                          sweepData[angle] = int(np.random.randint(0,totalDistance,1));
                        sleep(0.04);

# Allow time for initial sweep..i.e the bot is started and takes a initial sweep to provide guidance
time.sleep(25);


# Keep checking for unsafe points. Call neural network module if unsafe point is found.
while True:
                if obstacleDetected == True :
                    callNeuralNetwork()
                
                print("\n**************Anti-Clock wise Rotation-*************")
                leftrotate()
                thread.start_new_thread(lidarGetDistance,(0,leftAngle)); # starting get distance thread           
                time.sleep(3)
                stop()
                
                print("\n**************Clock wise Rotation-*************")
                rightrotate()
                thread.start_new_thread(lidarGetDistance,(leftAngle,0));
                time.sleep(3)
                stop()
                
                if obstacleDetected == True :
                    callNeuralNetwork()
                
                rightrotate()
                thread.start_new_thread(lidarGetDistance,(0,rightAngle));
                time.sleep(3)
                stop()
                
                print("\n**************Anti-Clock wise Rotation-*************")
                leftrotate()
                thread.start_new_thread(lidarGetDistance,(rightAngle,0)); # starting get distance thread           
                time.sleep(3)
                stop()
                

