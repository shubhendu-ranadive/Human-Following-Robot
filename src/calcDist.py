# import the necessary packages
import numpy as np
import cv2
import imutils
import argparse
import time

# define the lower and upper boundaries of the "green"
# box in the HSV color space
# blue (50,100,100) to (130.255,255)
# pink (110,0,100) to (190,255,255)
# light green (29,86,6) to (64,255,255)
# dark green (50,86,6) to (64,255,255)
# red(wall) (125,40,100) to (180,150,255)
greenLower = (50,100,100)
greenUpper = (130,255, 255)


def find_marker(frame):
	# convert the image to grayscale, blur it, and detect edges
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#gray = cv2.GaussianBlur(image, (5, 5), 0)
	#edged = cv2.Canny(gray, 35, 125)
        # resize the frame, blur it, and convert it to the HSV color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=4)
	mask = cv2.dilate(mask, None, iterations=4)
	#gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(mask, (5, 5), 0)
	cv2.imshow("gray",gray)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	(_,cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.5
KNOWN_HEIGHT = 8.0 
AREA = KNOWN_WIDTH*KNOWN_HEIGHT 
# initialize the list of images that we'll be using
#IMAGE_PATHS = ["img1.jpg","img2.jpg"]

# load the first image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("img4.jpg")
marker = find_marker(image)
focalLength = (marker[1][0]*marker[1][1] * KNOWN_DISTANCE) / AREA
box = np.int0(cv2.boxPoints(marker))
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
cv2.putText(image, "%.2fft" % (focalLength / 12),(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
# show the frame and record if the user presses a key
cv2.imshow("Frame", image)
key = cv2.waitKey(0)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
args = vars(ap.parse_args())

# if the video path was not supplied, grab the reference to the
# camera
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# loop over the frames
while True:
        # grab the current frame
	(grabbed, frame) = camera.read() 
	# check to see if we have reached the end of the video
	if not grabbed:
		break

	#image = cv2.imread(frame)
	marker = find_marker(frame)
	area = marker[1][0]*marker[1][1]
	inches = distance_to_camera(AREA, focalLength, area)

	# draw a bounding box around the image and display it
	box = np.int0(cv2.boxPoints(marker))
	cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
	cv2.putText(frame, "%.2fft" % (inches / 12),
		(frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	
	# show the frame and record if the user presses a key
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

