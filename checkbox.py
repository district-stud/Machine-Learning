# import libraries
#import os
import cv2
import numpy as np


# reading image
image_array = cv2.imread('img-1.jpg')

# check array type
type(image_array)

# output: numpy.ndarray

# convering image to gray scale
gray_scale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

# image thresholding

_, img_bin = cv2.threshold(gray_scale_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

img_bin = 255 - img_bin

from PIL import Image
from numpy import *

Image.fromarray(img_bin).show()

# set min width to detect horizontal lines
line_min_width = 15

# kernel to detect horizontal lines
kernal_h = np.ones((1,line_min_width), np.uint8)

# kernel to detect vertical lines
kernal_v = np.ones((line_min_width,1), np.uint8)

# horizontal kernel on the image
img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)

# verical kernel on the image
img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)


# combining the image

img_bin_final=img_bin_h|img_bin_v

_, labels, stats,_ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

for x,y,w,h,area in stats[2:]:
    cv2.rectangle(image_array,(x,y),(x+w,y+h),(0,255,0),2)

Image.fromarray(image_array).show()

#Attempt 1 to remove surrounding noise
#def is_contour_bad(c):
	# approximate the contour
	#peri = cv2.arcLength(c, True)
	#approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# the contour is 'bad' if it is not a rectangle
	#return not len(approx) == 4

# Importing ImageDraw for floodfill
from PIL import Image, ImageDraw
  
# Opening the image and changing type
img = Image.open("img-1.jpg").convert('RGBA')
  
# Location of seed
seed = (0, 0)
  
# Pixel Value which would be used
rep_value = (0, 0, 0, 0)
  
# Calling the floodfill() function and parameters
ImageDraw.floodfill(img, seed, rep_value, thresh = 100)
  
img.show()

from PIL import Image, ImageDraw

#Trying to mask everything except checkbox using contours 
def mask(image_array, contours_region):
    #cover everything but the square
    stencil = np.zeros(image_array.shape).astype(image_array.dtype)
    #contours = contours_region
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours_region, color)
    image = cv2.bitwise_and(image_array, stencil)
    #cv2.imshow('a',image)
    return image
#cv2.imshow('a',image_array)

image_array.show()

