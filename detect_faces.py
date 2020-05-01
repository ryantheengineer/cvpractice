# Original tutorial URL: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
# Required arguments: --image (path to input image), --prototxt (path to Caffe
# prototxt file), --model (path to pretrained Caffe model)
# Optional argument: --confidence (overwrites the default threshold of 0.5)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network adn obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    
