# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:13:30 2019

@author: pruthvi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:42:32 2019

@author: pruthvi
"""

#%%

#%%
        
 # import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in mm)")
#ap.add_argument("-m", "--metric", type=float, required=True,
#    help="how many pixel are in one inch / cm. Supply 1 if you want the object size in units of pixel")
args = vars(ap.parse_args())


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
    
while True:
    # Capture frame-by-frame
    grabbed, frame = vs.read()  # ret = 1 if the video is captured; frame is the image
    
    if not grabbed:
        break
    
    if W is None or H is None:
        (H,W) = frame.shape[:2]
    
    start = time.time()
    end = time.time()
    
    # Our operations on the frame come here    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    edged=cv2.Canny(gray,50,100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    
    # loop over the contours individually
    for c in cnts:
	    # if the contour is not sufficiently large, ignore it
	    if cv2.contourArea(c) < 200:
		    continue

	    # compute the rotated bounding box of the contour
	    orig = frame.copy()
	    box = cv2.minAreaRect(c)
	    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	    box = np.array(box, dtype="int")

	    # order the points in the contour such that they appear
	    # in top-left, top-right, bottom-right, and bottom-left
	    # order, then draw the outline of the rotated bounding
	    # box
	    box = perspective.order_points(box)
	    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	    # loop over the original points and draw them
	    for (x, y) in box:
		    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	    # unpack the ordered bounding box, then compute the midpoint
	    # between the top-left and top-right coordinates, followed by
	    # the midpoint between bottom-left and bottom-right coordinates
	    (tl, tr, br, bl) = box
	    (tltrX, tltrY) = midpoint(tl, tr)
	    (blbrX, blbrY) = midpoint(bl, br)
    
	    # compute the midpoint between the top-left and top-right points,
	    # followed by the midpoint between the top-righ and bottom-right
	    (tlblX, tlblY) = midpoint(tl, bl)
	    (trbrX, trbrY) = midpoint(tr, br)
    
	    # draw the midpoints on the image
	    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    
	    # draw lines between the midpoints
	    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
	    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
    
	    # compute the Euclidean distance between the midpoints
	    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
	    # if the pixels per metric has not been initialized, then
	    # compute it as the ratio of pixels to supplied metric
	    # (in this case, inches)
	    if pixelsPerMetric is None:
		    pixelsPerMetric = dB / args["width"] # args["width"]
            
	    # compute the size of the object
	    dimA = dA / pixelsPerMetric
	    dimB = dB / pixelsPerMetric
    
	    # draw the object sizes on the image
	    cv2.putText(orig, "{:.1f}mm".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
	    cv2.putText(orig, "{:.1f}mm".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    
#	    # show the output image
	    cv2.imshow("Image", orig)
#	    cv2.waitKey(0)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,(orig.shape[1], orig.shape[0]), True)
        
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
        
        
    writer.write(orig)
    
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
    



























