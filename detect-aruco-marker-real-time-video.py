# USAGE
# python detect-aruco-marker-real-time-video.py

# import packages
from imutils.video import VideoStream # for calling the video
import argparse # for argument processing
import imutils # for image resizing
import time # to set a sleep
import cv2 # to bind opencv
import sys # in case we need to exit

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-t","--type", type=str,
                default="DICT_5X5_100",
                help="type of the ArUco marker to detect")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# check if the -t supplied exists by passing it through the dictionary
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUco tag of '{}' is not supported".format(
		args["Type"]))
	sys.exit(0)

# load the ArUco dictionary and grab the parameters
print("[INFO] detecting '{}' tags ...".format(args["type"]))
# grab the dictionary of ArUco markers
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
# define the detection parameters () == default
arucoParams = cv2.aruco.DetectorParameters_create()

# initialize the video stream and allow camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the grames from the video stream
while True:
	# grab the frome from the threaded video stream and resize to width=100 for play stream
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	# detect ArUco markers in the input frame
	(corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
		# image = input image/frame, arucoDict we're using, parameteres == default
							arucoDict, parameters=arucoParams)
		# this returns 3 values: corners, ids, rejected

	# verify at LEAST 1 ArUco marker was detected
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()

	# loop over the detected corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners, which are always in order
			# top-left
			# top-right
			# bottom-right
			# bottom-left
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# convert each x, y coordinate pair to integers
			topLeft = (int(topLeft[0]), int(topLeft[1]))
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

			# draw bounding box
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

			# compute and draw center x, y of the detected marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

			# draw the ArUco marker id on the frame
			cv2.putText(frame, str(markerID),
						(topLeft[0], topLeft[1] - 15),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (0, 255, 0), 2)

# --------REUSABLE FOR VIDEO CV------

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()