from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import pytesseract
import imutils
import time
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\bob65\AppData\Local\Tesseract-OCR\tesseract.exe"

iteration = 0;

def interpretEastOutput(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# If EAST is less confident than the minimum level spcified,
			# ignore the reading
			if scoresData[x] < eastConfidence:
				continue

			# Compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# Extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# Use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# Compute the start and end coordinates of the bounding boxes,
			# sines and cosines are used to ensure that bounding box fully
			# contains angled text
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# Note the boxes we calculated with their respective scores
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# Arguments for East and TesseractOCR models
eastFile = "frozen_east_text_detection.pb"
videoFile = "IMG_1007.MOV"

eastConfidence = .99
eastRescale = (320,320)
padding = 0.1

# Initialize frame dimensions
(W, H) = (None, None)
(newW, newH) = eastRescale
(ratioW, ratioH) = (None, None)

# Define output layers of EAST model
layers = [ "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# Load in EAST model to CV2
print("[INFO] fetching EAST model...")
net = cv2.dnn.readNet(eastFile)

# Set reference to the video file
stream = cv2.VideoCapture(videoFile)

# Create video output object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(videoFile + '_Results.MOV', fourcc, 30, (1920, 1080))

# Create file writing object
file = open(videoFile + ".txt", "w")


# Loop over all frames in the Video File
while True:
	# Get current frame
	frame = stream.read()
	frame = frame[1]

	# Is video over?
	if frame is None:
		break

	# Resize the frame to a larger frame with the same aspec ratio
	frame = imutils.resize(frame, width = 1920)
	original = frame.copy()
	# Pass :2 to the shape method to ignore the color channel
	(origH, origW) = original.shape[:2]

	(H, W) = frame.shape[:2]
	ratioH = H / float(newH)
	ratioW = W / float(newW)

	# Resize the frame based on the passed rescale factors
	frame = cv2.resize(frame, eastRescale)

	# Reformat the frame into a shape blob Matrix acceptable by the input layer of EAST
	blob = cv2.dnn.blobFromImage(frame, 1.0, eastRescale, (123.68, 116.78, 103.94),
								 swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layers)

	# Interpret output of the EAST model, apply non-max supression in order to
	# suppress the weak overlapping bounding boxes
	(rects, confidences) = interpretEastOutput(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	
    # Keep track of the current frame
	iteration = iteration + 1

	# Loop over the bounding boxes to pass them to the tesseract OCR
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * ratioW)
		startY = int(startY * ratioH)
		endX = int(endX * ratioW)
		endY = int(endY * ratioH)

		# Apply padding chosen by user in order to ensure edges of text are
		# not truncated by EAST boxes
		dX = int((endX - startX) * padding)
		dY = int((endY - startY) * padding)

		# Apply the padding to both sides of the of bounding box
		# Choose the max of the padded starting and 0; this ensures that the
		# padded bounding box does not index outside of the original image
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)

		# Choose the min of the padded ending and the original outside edges
		# of the image; this ensures that the bounding box does not index
		# original image
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		# Get region of interest for the Tesseract OCR to search
		ROI = original[startY:endY, startX:endX]

		# in order to apply Tesseract v4 to OCR text we must supply
		# (1) a language, (2) an OEM flag of 4, indicating that the we
		# wish to use the LSTM neural net model for OCR, and finally
		# (3) an OEM value, in this case, 7 which implies that we are
		# treating the ROI as a single line of text
		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(ROI, config=config)
        
        # Print text and bounding boxes to image
		cv2.rectangle(original, (startX, startY), (endX, endY), (255, 0, 0), 2)
		cv2.putText(original, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX,
					1.2, (255, 0, 0), 3)
        
        # If there is text within the bounding box, write it to the output file
		if text != "":
			file.write("Frame " + str(iteration) + ": [" + str(startX) + ", " + str(startY) + ", " + str(endX) +", " + str(endY) + "], " + text + "\n")

	out.write(original)

	# show the output frame
	cv2.imshow("Text Detection", original)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# Drop pointer to video file, and file
stream.release()
out.release()
file.close()

# Destroy all windows
cv2.destroyAllWindows()