#!/usr/bin/env python

"""espotter_filter.py: Nothing to know at this point"""

__author__      = "w3llschmidt@gmail.com"
__copyright__   = "Copyright 2018, Planet Earth"

__license__ = "GPL"
__version__ = "0.1"
__status__ = "DONT USE!"

################################################################################################################################

import cv2
import numpy

################################################################################################################################

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = numpy.array([((i / 255.0) ** invGamma) * 255
    for i in numpy.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

################################################################################################################################

bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

################################################################################################################################

class MyFilter:
    
    def process(self, frame):

    while(1):

        r = (670, 2, 829, 861)
        frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 5)
        blurred = adjust_gamma(blurred, 5)

        fgmask = bgSubtractor.apply(blurred)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        dilation = cv2.dilate(fgmask, kernel, iterations=5)

        img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)

            if radius >= 10:

                cv2.circle(frame,center,radius,(0,255,0),3,8)

                position = (center[0]-10, center[1]-18)

            else:
                continue    

        return frame
        
def init_filter():

    f = MyFilter()
    return f.process
