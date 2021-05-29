
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations
import pyaudio
import wave
import threading


def calculateDistance(point1, point2):
    """
    Use to detect Social Distancing.
    Using Euclidean Distance method to get the object distance in the video
    For every frame the program will calculate for Euclidean Distance.

    This type of computation will need a proper distance of camera 
    but for the purpose of presentation the Condition for Euclidean will set to proper distance.

    """
    euclidean_distance = math.sqrt(point1**2 + point2**2)
    return euclidean_distance 


def getRectangleCoordinates(x, y, w, h): 
    """
    To get the center coordinates then convert rectangle
    to ensure precision of rectangle in a person
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def play_audio_file():
    """
    To play a record audio.
    Advicing the people to follow Social Distancing
    """
    global threadStatus
    ding_wav = wave.open('./Paging.wav', 'rb')
    ding_data = ding_wav.readframes(ding_wav.getnframes())
    audio = pyaudio.PyAudio()
    stream_out = audio.open(
        format=audio.get_format_from_width(ding_wav.getsampwidth()),
        channels=ding_wav.getnchannels(),
        rate=ding_wav.getframerate(), input=False, output=True)
    stream_out.start_stream()
    stream_out.write(ding_data)
    time.sleep(0.2)
    threadStatus = False 
    stream_out.stop_stream()
    stream_out.close()
    audio.terminate() 

threadStatus = False

def drawRectangle(object, img):
    """
    Filter Person in a list of object in a video frame.
    """
    camDistance = 1

    if len(object) > 0:  						
        centroid_dict = dict() 						
        objectId = 0								
        for label, confidence, bbox in object:	
            if label == 'person':                
                x, y, w, h = (bbox[0], bbox[1], bbox[2], bbox[3])     	
                xmin, ymin, xmax, ymax = getRectangleCoordinates(float(x), float(y), float(w), float(h))  # get a coordinates of rectangle      
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Adding center point for every person detected, each person will have its on unique id
                objectId += 1 # increment ID per every detected object.
              	
        red_zone_list = [] # List containing which Object id is in under threshold distance condition. 
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get id and point of two object
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	# get the difference between centroid x: 0, y :1
            distance = calculateDistance(dx, dy) 	# call function in line 15 to calculate the Euclidean Distance

            if camDistance == 2:   # Camera distance for presentation
                compare = 520
            else:                  # For a wide area Monitoring
                compare = 75

            # print(str(distance) + ' Euclidean Distance') 
            if distance < compare:					# check if the objects is violating the Social Distancing. condition 'compare' is dependent to its camera distance.
                global threadStatus
                if(threadStatus == False):
                    threadStatus = True
                    audioThread = threading.Thread(target=play_audio_file,args=())
                    audioThread.start()
                    
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)       #  Add object ID to a violation list 1st object
                    red_line_list.append(p1[0:2])   #  Add points to the list 1st object
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)		#  Add object ID to a violation list 2nd object 
                    red_line_list.append(p2[0:2])   #  Add points to the list 2nd object
        
        for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in red_zone_list:   # if id is in red zone list
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) # Draw red rectangle for violated object
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # CDraw green rectangle

        text = "People at Risk: %s" % str(len(red_zone_list)) 			# Display number of people at risk
        location = (15,30)											
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # adding text to open cv

        for check in range(0, len(red_line_list)-1):					# Draw a line red line between two or more object.
            start_point = red_line_list[check]                          
            end_point = red_line_list[check+1]
            check_line_x = abs(end_point[0] - start_point[0])   		# get the coordinates for x axis
            check_line_y = abs(end_point[1] - start_point[1])			# get the coordinates for y axis
            if (check_line_x < 75) and (check_line_y < 25):				# check if social distance is violated then draw the a red line between the two object if true
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)   
    return img


netMain = None
metaMain = None
altNames = None



def TU_DISTANCIA():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    network, class_names, class_colors = darknet.load_network(configPath,  metaPath, weightPath, batch_size=1)
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("pedestrians.mp4")
    width = int(cap.get(3))
    height = int(cap.get(4))
    new_height, new_width =  800 , 1200 # Use for Live video
    # new_height, new_width =  height // 2 , width // 2 # use for recorded video

    # Save the capture video.
    out = cv2.VideoWriter("./Demo/pedestrian_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,(new_width, new_height))
    

    # For each detect create an image.
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()

        if not ret: #if frame is not present or no return end the task
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,(new_width, new_height), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)
        image = drawRectangle(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Tu Distancia', image)
        key = cv2.waitKey(1)
        if key == ord('q'): 
            break
        out.write(image)

    cap.release()
    out.release()
    print("Tu Distancia :: Video has been Save") #Console print if video has complete recorded.

if __name__ == "__main__":
    TU_DISTANCIA()