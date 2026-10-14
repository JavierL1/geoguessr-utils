import cv2
import numpy as np


def detect_digit_geometrically(image_path):
    # 1. Read the image in grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return "Error: Could not load image."

    # 2. Threshold (Make digit white, background black)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 3. Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours or hierarchy is None:
        return "Empty"

    # 4. Find the largest contour (the digit itself)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if h == 0 or w == 0:
        return "Empty"

    # 5. Check for holes -> It's a '0'
    has_hole = False
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:  # Has a parent contour, meaning it's a hole
            if cv2.contourArea(contour) > 20:
                has_hole = True
                break

    if has_hole:
        return "0"

    # 6. Differentiate between '1' and '2' using Bottom Width
    # Look at the bottom 30% of the bounding box
    bottom_start_y = y + int(h * 0.7)
    bottom_slice = thresh[bottom_start_y : y + h, x : x + w]

    # Find the columns (x-coordinates) that contain white pixels in this bottom slice
    cols_with_white = np.where(bottom_slice == 255)[1]

    if len(cols_with_white) == 0:
        # Fallback just in case the bottom slice was empty for some reason
        return "1"

    # Calculate how wide the bottom is compared to the total width of the bounding box
    bottom_width = cols_with_white[-1] - cols_with_white[0]
    bottom_ratio = bottom_width / float(w)

    # Print for debugging
    print(
        f"Digit Bounding Box: {w}x{h} | Bottom Width: {bottom_width} | Bottom Ratio: {bottom_ratio:.2f}"
    )

    # A '1' has a very narrow bottom (ratio usually < 0.4)
    # A '2' has a wide flat base (ratio usually > 0.7)
    if bottom_ratio < 0.5:
        return "1"
    else:
        return "2"


# --- TEST IT ---
if __name__ == "__main__":
    # Replace with the path to your image
    # (Use the exact image you sent me: the black 0 on white background)
    result = detect_digit_geometrically("overall_score_0.jpg")

    print(f"Detected Digit: {result}")

    result = detect_digit_geometrically("overall_score_1.jpg")

    print(f"Detected Digit: {result}")
    result = detect_digit_geometrically("overall_score_2.jpg")

    print(f"Detected Digit: {result}")
