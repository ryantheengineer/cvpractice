"""
    A computer vision project to calculate the % moon phase in an image.
"""

# import the necessary packages
import numpy as np
import cv2
import imutils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to moon image")
args = vars(ap.parse_args())

# Import image
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

# Blur the grayscale image
blur = cv2.GaussianBlur(gray, (11, 11), 0)
cv2.imshow("Blurred", blur)
cv2.waitKey(0)

# Detect moon in image and create mask
thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# Erode and dilate the threshold to smooth the shape
mask = thresh.copy()
print(mask)
# mask = cv2.erode(mask, None, iterations=5)
# mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Erode and Dilate", mask)
cv2.waitKey(0)

# Get pixel count of lit section of moon
lit = np.count_nonzero(mask)
print(lit)

# Fit circle to the moon mask
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
center = None

if len(cnts) > 0:
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # only proceed if the radius meets a minimum size
    if radius > 10:
        # draw the circle and centroid on the picture and the threshold image
        cv2.circle(image, (int(x), int(y)), int(radius),
            (0, 255, 255), 2)

else:
    print("[ERROR] No moon contour detected")

cv2.imshow("Moon with detected circle", image)
cv2.waitKey(0)

# Create a mask from the circle determined above
cmask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(cmask, (int(x), int(y)), int(radius), 255, -1)

# Count the pixels in the circle
moon = np.count_nonzero(cmask)
print(moon)

# Calculate the percentage of "lit" pixels that are within the circular area
percentage = (float(lit) / float(moon)) * 100
print("{:.2f}% Illuminated moon".format(percentage))


# Display original image with annotation showing the detected moon and the
# phase percentage
