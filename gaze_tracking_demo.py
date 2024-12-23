import cv2
import numpy as np
import dlib
from math import hypot
import matplotlib.pyplot as plt

# Initialize video capture
cap = cv2.VideoCapture(1)

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Font for displaying text
font = cv2.FONT_HERSHEY_PLAIN

# List to store pupil coordinates
pupil_coords = []

# Screen dimensions
screen_width, screen_height = 1920,1080

# Helper function to calculate the midpoint
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to map coordinates to screen resolution
def map_to_screen(pupil_x, pupil_y, frame_width, frame_height):
    screen_x = int(pupil_x * screen_width / frame_width)
    screen_y = int(pupil_y * screen_height / frame_height)
    return screen_x, screen_y

# Function to detect pupil center
def get_pupil_center(eye_points, facial_landmarks, gray, scale=0.6):
    # Get the bounding box of the eye region
    eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
    x, y, w, h = cv2.boundingRect(eye_region)

    # Shrink the bounding box around the eye region
    center_x, center_y = x + w // 2, y + h // 2
    w, h = int(w * scale), int(h * scale)
    x, y = center_x - w // 2, center_y - h // 2

    # Crop the region of interest (ROI)
    eye_roi = gray[y:y + h, x:x + w]
    eye_roi = cv2.equalizeHist(eye_roi)

    # Thresholding to detect dark areas (pupils)
    _, threshold_eye = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        return int(cx + x), int(cy + y)  # Convert to original frame coordinates
    return None

# Main loop
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Get pupil centers
        left_pupil = get_pupil_center([36, 37, 38, 39, 40, 41], landmarks, gray, scale=0.6)
        right_pupil = get_pupil_center([42, 43, 44, 45, 46, 47], landmarks, gray, scale=0.6)

        # Display pupil coordinates
        if left_pupil:
            cv2.circle(frame, left_pupil, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Left: {left_pupil}", (10, 30), font, 1, (255, 255, 255), 1)
            pupil_coords.append(left_pupil)

            # Map to screen coordinates
            screen_left_pupil = map_to_screen(left_pupil[0], left_pupil[1], frame.shape[1], frame.shape[0])
            cv2.putText(frame, f"Screen Left: {screen_left_pupil}", (10, 50), font, 1, (255, 255, 0), 1)

        if right_pupil:
            cv2.circle(frame, right_pupil, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Right: {right_pupil}", (10, 60), font, 1, (255, 255, 255), 1)
            pupil_coords.append(right_pupil)

            # Map to screen coordinates
            screen_right_pupil = map_to_screen(right_pupil[0], right_pupil[1], frame.shape[1], frame.shape[0])
            cv2.putText(frame, f"Screen Right: {screen_right_pupil}", (10, 80), font, 1, (255, 255, 0), 1)

    # Show frame
    cv2.putText(frame, "Press ESC to exit", (10, frame.shape[0] - 10), font, 1, (255, 255, 255), 1)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Escape key
        break

# Release resources
cap.release()
cv2.destroyWindow("Frame")

# Generate heatmap
if pupil_coords:
    heatmap, xedges, yedges = np.histogram2d(
        [coord[0] for coord in pupil_coords],
        [coord[1] for coord in pupil_coords],
        bins=(frame.shape[1] // 10, frame.shape[0] // 10),
        range=[[0, frame.shape[1]], [0, frame.shape[0]]]
    )

    plt.imshow(heatmap.T, origin='lower', cmap='hot', extent=[0, frame.shape[1], 0, frame.shape[0]])
    plt.colorbar(label="Focus Intensity")
    plt.title("Eye Focus Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("eye_focus_heatmap.png")
    plt.show()

    # Save pupil coordinates to file
    np.savetxt("pupil_coords.csv", pupil_coords, delimiter=",", fmt="%d", header="x,y")
