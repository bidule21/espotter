
#!/usr/bin/env python

"""espotter.py: Nothing to know at this point"""

__author__      = "w3llschmidt@gmail.com"
__copyright__   = "Copyright 20018, Planet Earth"

__license__ = "GPL"
__version__ = "0.1"
__status__ = "Alpha"

################################################################################################################################

from time import sleep

import numpy as np
import copy
import cv2

################################################################################################################################

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
	for i in np.arange(0, 256)]).astype("uint8")
 	return cv2.LUT(image, table)

################################################################################################################################

# Initialize the video // here we will have the raspi 8MP livefeed in the release version
cap = cv2.VideoCapture('scheibe.mp4')

################################################################################################################################

# https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html?highlight=createbackgroundsubtractormog#backgroundsubtractormog
# Create the background subtraction object
bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

# Create the kernel that will be used to remove the noise in the foreground mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

################################################################################################################################

def main():

	framecount = 0
	shotcount = 0

	font = cv2.FONT_HERSHEY_SIMPLEX

	while(1):

		# Get the next frame
		ret, frame = cap.read()

		# End here if there are no (more) frames
		if ret == False:
			print str("Could not open feed!")
			break

		#r = cv2.selectROI(frame)
		#imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
		#cv2.imshow("Image", imCrop)
		#print r
		#cv2.waitKey(0)

		feed = copy.copy(frame)
		feed = cv2.resize(feed, (1024/2,768/2))
		cv2.namedWindow("feed")
		cv2.moveWindow("feed", 0,0);
		cv2.imshow("feed",feed)

		# Set ROI // region of intrest
		r = (670, 2, 829, 861)
		frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

		# Convert frame to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Blurring the grayscale frame and adjust gamma
		blurred = cv2.GaussianBlur(gray, (5, 5), 5)
		blurred = adjust_gamma(blurred, 5)

		monitor_blurred = copy.copy(blurred)
		monitor_blurred = cv2.resize(monitor_blurred, (1024/2,768/2))
		cv2.namedWindow("monitor_blurred")
		cv2.moveWindow("monitor_blurred", 512,0);
		cv2.imshow("monitor_blurred",monitor_blurred)

		# https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html?highlight=createbackgroundsubtractormog#backgroundsubtractor
		# Obtain the foreground mask
		fgmask = bgSubtractor.apply(blurred)

		# Remove part of the noise
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

		monitor_fgmask = copy.copy(fgmask)
		monitor_fgmask = cv2.resize(monitor_fgmask, (1024/2,768/2))
		cv2.namedWindow("monitor_fgmask")
		cv2.moveWindow("monitor_fgmask", 1024,0);
		cv2.imshow("monitor_fgmask",monitor_fgmask)

		# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
		# We dont want erosion at this point as we have adjusted the gamma
		dilation = cv2.dilate(fgmask, kernel, iterations=5)

		monitor_dilation = copy.copy(dilation)
		monitor_dilation = cv2.resize(monitor_dilation, (1024/2,768/2))
		cv2.namedWindow("monitor_dilation")
		cv2.moveWindow("monitor_dilation", 1536,0);
		cv2.imshow("monitor_dilation",monitor_dilation)

		# finding external contours (shot)
		cnt_frame, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# for each contour plot a circle
		for contour in contours:
			(x,y),radius = cv2.minEnclosingCircle(contour)
			center = (int(x),int(y))
			radius = int(radius)

			# Limit to contours equal/larger then a shot
			if radius >= 10:
				cv2.circle(frame,center,radius,(0,255,0),3,8)
				# Give some time to look at
				sleep(0.1)
			else:
				continue	

			# Shotcounter
			# shotcount +=1
			# cv2.putText(frame, str(shotcount), (1780, 80), font, 2.0, (0, 255, 0), 2, cv2.LINE_AA)

		# Framecounter	
		framecount +=1
		cv2.putText(frame, str(framecount), (10, 30), font, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

		frame = cv2.resize(frame, (1024,768)) # http://www.iosres.com/index-legacy.html
		cv2.namedWindow("frame")
		cv2.moveWindow("frame", 000,450);
		cv2.imshow("frame",frame)

		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

# this is the standard boilerplate that calls the main() function
if __name__ == '__main__':
    # sys.exit(main(sys.argv)) # used to give a better look to exists
	main()