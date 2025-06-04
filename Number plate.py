import cv2
import imutils
import pytesseract

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
image = cv2.imread('car3.jpg')
if image is None:
    print("Error: Unable to load image. Check the file path.")
    exit()

# Resize the image for better processing
image = imutils.resize(image, width=600)
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)

# Apply bilateral filtering to smooth the image
smooth = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Smoothed Image", smooth)
cv2.waitKey(0)

# Detect edges using Canny edge detection
corner = cv2.Canny(smooth, 170, 200)
cv2.imshow("Highlighted Edges", corner)
cv2.waitKey(0)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(corner.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours on a copy of the original image
image1 = image.copy()
cv2.drawContours(image1, contours, -1, (0, 0, 255), 3)
cv2.imshow('Edge Segmentation', image1)
cv2.waitKey(0)

# Sort contours by area and keep the largest 30
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
NoPlate = None

# Draw the top 30 contours
image2 = image.copy()
cv2.drawContours(image2, contours, -1, (0, 255, 0), 3)
cv2.imshow("Number Plate Segmentation", image2)
cv2.waitKey(0)

# Loop through contours to find a rectangular contour (number plate)
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:  # Look for a rectangular contour
        NoPlate = approx
        x, y, w, h = cv2.boundingRect(contour)
        crp_image = image[y:y + h, x:x + w]

        # Save the cropped number plate image
        cv2.imwrite('NumberPlate.png', crp_image)

        # Display the cropped number plate
        cv2.imshow("Cropped Number Plate", crp_image)
        cv2.waitKey(0)
        break

# Highlight the detected number plate on the original image
if NoPlate is not None:
    cv2.drawContours(image, [NoPlate], -1, (0, 255, 0), 3)
    cv2.imshow("Final Image with Number Plate", image)
    cv2.waitKey(0)

    # Use Tesseract OCR to extract text from the number plate
    text = pytesseract.image_to_string(crp_image, config='--psm 8')
    print(f"Detected Number Plate: {text.strip()}")
else:
    print("Number plate not detected.")

cv2.destroyAllWindows()