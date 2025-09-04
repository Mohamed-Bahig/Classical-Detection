import cv2
import numpy as np

def detect_shape(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        shape = "Square" 
    else:
        # Use circularity check
        area = cv2.contourArea(c)
        if peri == 0:
            shape = "unidentified"
        else:
            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity > 0.8:
                shape = "Circle"
            else:
                shape = "X Shape"

    return shape, vertices

def main():
    # Load image
    img = cv2.imread(r"E:\Collage\MIA\Task 11\Task 11.2\test_image.jpg")
    if img is None:
        raise FileNotFoundError("Image not found. Check path and filename.")

    output = img.copy()
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Invert threshold (since shapes are black, background white)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 150 or cv2.contourArea(c) > 700:  # Ignore tiny noise
            continue

        shape, vertices = detect_shape(c)

        # Get centroid
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Draw contour + label
        cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
        cv2.putText(output, f"{shape} ({vertices})", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite(r"E:\Collage\MIA\Task 11\Task 11.2\output.jpg", output)
    cv2.imshow("Detected Shapes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
