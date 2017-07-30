import cv2
import sys
import dlib
import numpy as np
import math
from time import time
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
# HSV Thresh bounds(Tweak these for better results) - Currently using BLUE
#BLUE
upper_bound = np.array([110, 256, 256])
lower_bound = np.array([94, 122, 45])



#Variables that need to be stored across frames
drawing_points = [(0, 0)]
prev_pt = (-1, -1)
velocity = 0
velocity_ticker = 0
t_s = -1
height = 0  # 480
width = 0  # 640
HIT_START = False

# Defining the kernel for the morphological transformations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))



def drawPoints(img, points):
    n = len(points)
    for r, point in enumerate(points[0:n-2]):
        cv2.line(img, point, points[r+1], (255, 0, 0), thickness= 10)

    # for point in points:
    #     cv2.circle(img, point, 6,  (0, 0, 255), -1)
    return img
def calcVelocity(pt1, pt2):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    return int(math.sqrt(dy*dy + dx*dx))
def smallestBB(points, w, h):
    x_max = 0
    y_max = 0
    x_min = w-1
    y_min = h-1
    for point in points:
        x = point[0]
        y = point[1]
        if x >= x_max:
            x_max = x
        if x <= x_min:
            x_min = x
        if y >= y_max:
            y_max = y
        if y <= y_min:
            y_min = y
    return [(x_min, y_min), (x_max, y_max)]
def generateDigitImage(bbox, points):
    global height
    global width
    x_max = bbox[1][0]
    x_min = bbox[0][0]
    y_max = bbox[1][1]
    y_min = bbox[0][1]
    img = np.zeros((width, height))
    img = drawPoints(img, points)
    #
    return img
def get_tfl_model():
    network = input_data(shape=[None, 28, 28, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(network=network)
    model.load('mnist_model.tfl')
    return model
def get_contours(img):
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Thresholds based on predetermined bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Operations on the Binary Mask to reomve noise and highlight the foreground
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel=kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel=kernel)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Working with contours and isolating the largest contour
    mask_c = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel=kernel)
    contours, _ = cv2.findContours(mask_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return mask_c, contours
def get_largest_contour(contours):
    largest_contour = 0
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) >= cv2.contourArea(contours[largest_contour]):
            largest_contour = i
    return largest_contour
def get_centroid(mask_c):
    M = cv2.moments(mask_c, True)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx = -1
        cy = -1
    return (cx, cy)

model = get_tfl_model()
cam = cv2.VideoCapture(0)
while True:
    found = False
    ret, img = cam.read()
    height, width, channels = img.shape
    img = cv2.flip(img, 1)
    if t_s != -1:
        if time() - t_s>10:
            cv2.destroyWindow("Digit")
            t_s = -1

    mask_c, contours = get_contours(img)
    largest_contour = get_largest_contour(contours)
    cv2.drawContours(mask_c, contours, largest_contour, (255, 255, 255), thickness=-200)

    # Calculating and Drawing Centroid of Shape
    M = cv2.moments(mask_c, True)
    cx, cy = get_centroid(mask_c)
    if cx != -1 and cy != -1:
        # Adding current point to Drawing
        drawing_points.append((cx, cy))
        # Velocity
        if prev_pt != (-1, -1):
            velocity = calcVelocity(prev_pt, (cx, cy))
        else:
            velocity = 0
        prev_pt = (cx, cy)
    else:
        velocity = 0

    # Adding points and lines of the Drawing
    img = drawPoints(img, drawing_points[1:])

    # Finding the smallest BB that fits the Drawing and drawing it on the image
    bbox = smallestBB(drawing_points[1:], width, height)
    cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), thickness=4)

    # Displaying the images
    cv2.imshow("Original", img)
    cv2.imshow("Binary Mask- After Processing", mask_c)

    # Logic using velocity to clear drawing
    if velocity < 7:
        velocity_ticker += 1
    if velocity_ticker == 5:
        digit_image = generateDigitImage(bbox, drawing_points[1:])
        n_pts = len(drawing_points)
        drawing_points = [(0, 0)]       # Clear Drawing
        velocity_ticker = 0
        if n_pts > 6 and digit_image.shape[0] != 0:
            t_s = time()
            digit_image_small = cv2.resize(digit_image, (28, 28))
            pred = model.predict(np.reshape(digit_image_small, (1, 28, 28, 1)))
            pred = np.argmax(pred)
            print pred
            cv2.putText(digit_image, str(pred), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow("Digit", digit_image)

    # Detecting Key Presses
    k = cv2.waitKey(1)
    if k == 1048690:
        drawing_points = [(0, 0)]       # 'R' -  Clear Drawing
    if k == 1048603:                    # 'Esc' - Exit
        break
cv2.destroyAllWindows()


