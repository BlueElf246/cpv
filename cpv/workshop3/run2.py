import cv2
import numpy as np

# Define the Harris corner detection parameters
harris_params = {'block_size': 2, 'ksize': 3, 'k': 0.04}

# Define the Harris corner detection callback function
def harris_corner_detection_callback(x):
    harris_params['block_size'] = cv2.getTrackbarPos('Block size', 'Harris corner detection')
    harris_params['ksize'] = cv2.getTrackbarPos('Kernel size', 'Harris corner detection')
    harris_params['k'] = cv2.getTrackbarPos('K', 'Harris corner detection') / 100.0

    # Load the image and apply Harris corner detection
    img = cv2.imread('/Users/datle/Desktop/CPV/cpv/workshop3/img_data/Cars108.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, harris_params['block_size'], harris_params['ksize'], harris_params['k'])

    # Normalize the output to display it as an image
    dst_norm = np.empty_like(dst)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)

    # Draw circles around the corners on the original image
    img[dst_norm_scaled > 0.01 * dst_norm_scaled.max()] = [0, 0, 255]

    # Display the output image
    cv2.imshow('Harris corner detection', img)

# Load the image and create a window to display the results
img = cv2.imread('/Users/datle/Desktop/CPV/cpv/workshop3/img_data/Cars108.png')
cv2.namedWindow('Harris corner detection')

# Create trackbars to adjust the Harris corner detection parameters
cv2.createTrackbar('Block size', 'Harris corner detection', harris_params['block_size'], 10, harris_corner_detection_callback)
cv2.createTrackbar('Kernel size', 'Harris corner detection', harris_params['ksize'], 10, harris_corner_detection_callback)
cv2.createTrackbar('K', 'Harris corner detection', int(harris_params['k'] * 100), 100, harris_corner_detection_callback)

# Display the image and wait for a key press
cv2.imshow('Harris corner detection', img)
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
