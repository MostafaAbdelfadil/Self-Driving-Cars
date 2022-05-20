import numpy as np
import argparse
import time
import cv2
import os





def process(image):
	(H, W) = image.shape[:2]
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))