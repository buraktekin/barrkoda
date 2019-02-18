from pyzbar import pyzbar
from imutils.video import VideoStream
from imutils.video import FPS

import argparse as ap
import numpy as np
import datetime
import time
import imutils
import cv2

# argument parser
args_parser = ap.ArgumentParser()
args_parser.add_argument("-nf", "--frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
args_parser.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args_parser.add_argument("-o", "--output", type=str, default="barcodes.csv",
    help="path to CSV file keeping barcodes")   
args = vars(args_parser.parse_args())
# end of parser

# start streaming by the webcam (src=0 for webcam)
print("[INFO]:: starting video stream...")
video_stream = VideoStream(src=0).start()
time.sleep(2)
fps = FPS()

# open the output CSV file for writing and initialize the set of
# barcodes found thus far
csv = open(args["output"], "w")
found = set()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to
    # # have a maximum width of 400 pixels
    frame = video_stream.read()
    frame = imutils.resize(frame, width=600)
    
    # find the barcodes in the frame and decode each of the barcodes
    barcodes = pyzbar.decode(frame)

    for barcode in barcodes:
        points = barcode.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else: 
            hull = points
        
        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcode_content = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        
        # Number of points in the convex hull
        n = len(hull)

        # Draw the convext hull
        for j in range(0,n):
            cv2.line(frame, hull[j], hull[ (j+1) % n], (255,0,0), 3)

        print("[INFO]:: Barcode Type {}, Barcode Object: {}".format(barcode_type, barcode_content))

        # if the barcode text is currently not in our CSV file, write
		# the timestamp + barcode to disk and update the set
        if barcode_content not in found:
            csv.write("{},{}\n".format(datetime.datetime.now(), barcode_content))
            csv.flush()
            found.add(barcode_content)

    # show the output image
    # show the output frame
    cv2.imshow("Barcode Scanner", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# close the output CSV file do a bit of cleanup
print("[INFO] cleaning up...")
csv.close()
cv2.destroyAllWindows()
video_stream.stop()