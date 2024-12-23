import cv2
import numpy as np
import dlib
import threading
from math import hypot
import matplotlib.pyplot as plt

class EyeTracker(threading.Thread):
    def __init__(self, screen_width=1920, screen_height=1080, shape_predictor_path="shape_predictor_68_face_landmarks.dat"):
        super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.shape_predictor_path = shape_predictor_path
        self.cap = cv2.VideoCapture(1)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)
        self.pupil_coords = []
        self.running = False  # Flag to control the thread
        self.frame = None

    def midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    def map_to_screen(self, pupil_x, pupil_y, frame_width, frame_height):
        screen_x = int(pupil_x * self.screen_width / frame_width)
        screen_y = int(pupil_y * self.screen_height / frame_height)
        return screen_x, screen_y

    def get_pupil_center(self, eye_points, facial_landmarks, gray, scale=0.6):
        eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
        x, y, w, h = cv2.boundingRect(eye_region)
        center_x, center_y = x + w // 2, y + h // 2
        w, h = int(w * scale), int(h * scale)
        x, y = center_x - w // 2, center_y - h // 2
        eye_roi = gray[y:y + h, x:x + w]
        eye_roi = cv2.equalizeHist(eye_roi)
        _, threshold_eye = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            return int(cx + x), int(cy + y)
        return None

    def run(self):
        self.running = True
        while self.running:
            ret, self.frame = self.cap.read()
            if not ret or self.frame is None:
                print("Failed to grab frame. Exiting...")
                break
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            for face in faces:
                landmarks = self.predictor(gray, face)
                left_pupil = self.get_pupil_center([36, 37, 38, 39, 40, 41], landmarks, gray, scale=0.6)
                right_pupil = self.get_pupil_center([42, 43, 44, 45, 46, 47], landmarks, gray, scale=0.6)
                if left_pupil:
                    cv2.circle(self.frame, left_pupil, 5, (0, 255, 0), -1)
                    self.pupil_coords.append(left_pupil)
                if right_pupil:
                    cv2.circle(self.frame, right_pupil, 5, (0, 255, 0), -1)
                    self.pupil_coords.append(right_pupil)
            cv2.putText(self.frame, "Press ESC to exit", (10, self.frame.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                self.stop()
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.generate_heatmap()

    def generate_heatmap(self):
        if self.pupil_coords:
            heatmap, xedges, yedges = np.histogram2d(
                [coord[0] for coord in self.pupil_coords],
                [coord[1] for coord in self.pupil_coords],
                bins=(self.frame.shape[1] // 10, self.frame.shape[0] // 10),
                range=[[0, self.frame.shape[1]], [0, self.frame.shape[0]]]
            )
            plt.imshow(heatmap.T, origin='lower', cmap='hot', extent=[0, self.frame.shape[1], 0, self.frame.shape[0]])
            plt.colorbar(label="Focus Intensity")
            plt.title("Eye Focus Heatmap")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.savefig("eye_focus_heatmap.png")
            plt.show()
            np.savetxt("pupil_coords.csv", self.pupil_coords, delimiter=",", fmt="%d", header="x,y")
