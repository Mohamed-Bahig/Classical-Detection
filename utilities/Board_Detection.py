import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# Define the path to your dataset
mypath = '/home/rawan/Desktop/task11unlocked/dataset'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for filename in onlyfiles:
    img_path = join(mypath, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Could not read image: {img_path}")
        continue

    # Convert to grayscale and denoise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((20, 20), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Find largest contour
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea) if contours else None

    if max_contour is not None:
        # Create mask and crop
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        result = cv2.bitwise_and(image, image, mask=mask)

        x, y, w, h = cv2.boundingRect(max_contour)
        cropped = result[y:y+h, x:x+w]

        # Convert cropped image to grayscale
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(cropped_gray, (800, 600))

        # Show result
        cv2.imshow("Cropped & Grayscaled", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No contour found in image: {filename}")