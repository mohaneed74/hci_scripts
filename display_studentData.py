import customtkinter as ctk
import cv2
import mediapipe as mp
from PIL import Image
from Students_data import read_highschool_students_from_csv
from QuizGen import start_server_and_quiz
from learnChild import create_server_gui
from tkinter import messagebox
import time
import threading
from learnHandGesture import create_gesture
# Global variables to control the camera feed and gesture detection loop
stop_detection = False
start_tuio_quiz_button = None
start_gesture_quiz_button = None
option1 = False
option2 = False
app = None

# Function to detect hand gestures and trigger button clicks
def detect_gesture_and_click(student):
    global stop_detection, option1, option2, app

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    # Start webcam capture
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.9) as hands:
        start_time = None  # Timer to track 3-second window after option click
        
        while cap.isOpened():
            if stop_detection:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip and convert frame for processing
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks (skeleton of the hand)
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the coordinates for the fingers
                    index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_finger_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_finger_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    # Check if index finger is raised (Option 1)
                    if index_finger_tip.y < landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
                        middle_finger_tip.y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
                        ring_finger_tip.y > landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and \
                        pinky_finger_tip.y > landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
                        print("Option 1 clicked")
                        start_time = time.time()  # Start the 3-second timer
                        option1 = True

                    # Check if both index and middle fingers are raised (Option 2)
                    if index_finger_tip.y < landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
                        middle_finger_tip.y < landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
                        ring_finger_tip.y > landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and \
                        pinky_finger_tip.y > landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
                        print("Option 2 clicked")
                        start_time = time.time()  # Start the 3-second timer
                        option2 = True

            # If 3 seconds have passed since the option was clicked, stop the camera feed
            if start_time and time.time() - start_time >= 2:
                print("Closing camera after 2 seconds")
                if option1:
                    stop_detection = True  # Stop the gesture detection loop
                    cap.release()
                    cv2.destroyWindow("Hand Gesture Detection") 
                    app.after(0, lambda: [app.destroy(), create_server_gui(student)])
                    break
                    
                if option2:
                    stop_detection = True  # Stop the gesture detection loop
                    cap.release()
                    cv2.destroyWindow("Hand Gesture Detection")
                    app.after(0, lambda: [app.destroy(), create_gesture(student)])

                    break
            
            # Show the frame with detected hand landmarks
            cv2.imshow("Hand Gesture Detection", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources after camera feed is stopped
        cap.release()
        cv2.destroyWindow("Hand Gesture Detection")

# Read student data from CSV
student_data = read_highschool_students_from_csv('students data.csv')

def create_student_gui(student):
    global start_tuio_quiz_button, start_gesture_quiz_button, app

    # Initialize the application
    app = ctk.CTk()
    app.geometry("900x400+400+150")
    app.title("Student Information")

    # Main Frame
    main_frame = ctk.CTkFrame(app, width=900, height=600, corner_radius=10)
    main_frame.pack(padx=10, pady=10, fill="both", expand=True)

    # Left Frame for Image and Name
    left_frame = ctk.CTkFrame(main_frame, width=200, height=500, corner_radius=10)
    left_frame.grid(row=0, column=0, padx=30, pady=10, sticky="nsew")

    # Display Student Image
    try:
        image = Image.open(student['photo_path']).resize((180, 180))  # Resize the image
        student_image = ctk.CTkImage(image, size=(180, 180))
        image_label = ctk.CTkLabel(left_frame, image=student_image, text="")
        image_label.image = student_image  # Prevent garbage collection
        image_label.pack(pady=10)
    except Exception as e:
        print(f"Error loading image: {e}")
        image_label = ctk.CTkLabel(left_frame, text="Image Not Found")
        image_label.pack(pady=10)

    # Display Student Name
    name_label = ctk.CTkLabel(left_frame, text=student['name'], font=("Arial", 18, "bold"))
    name_label.pack()

    # Add Buttons for TUIO Quiz and Gesture Quiz
    start_tuio_quiz_button = ctk.CTkButton(
        left_frame, text="Start TUIO Learning",
        #command=lambda: [app.destroy(), create_server_gui(student)]  # Call TUIO Quiz Server
    )
    start_tuio_quiz_button.pack(pady=10)

    start_gesture_quiz_button = ctk.CTkButton(
        left_frame, text="Start Gesture Quiz",
        #command=lambda: [app.destroy(), start_gesture_quiz(student)]  # Function for Gesture Quiz
    )
    start_gesture_quiz_button.pack(pady=10)

    # Right Frame for Details and Images
    right_frame = ctk.CTkFrame(main_frame, width=600, height=500, corner_radius=10)
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    right_frame.grid_columnconfigure(0, weight=1)  # Configure column 0 for details
    right_frame.grid_columnconfigure(1, weight=1)  # Configure column 1 for images

    # Frames for Each Information Field (Column 0)
    fields = {
        "Student ID": student['student_id'],
        "Class": student['class'],
        "Grade": student['grade'],
        "Age": student['age'],
        "Gender": student['gender'],
        "Email": student['email'],
        "GPA": student['gpa'],
    }

    for i, (label, value) in enumerate(fields.items()):
        field_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        field_frame.grid(row=i, column=0, padx=10, pady=5, sticky="ew")

        field_label = ctk.CTkLabel(field_frame, text=f"{label}: ", font=("Arial", 14, "bold"))
        field_label.pack(side="left", padx=5)

        field_value = ctk.CTkLabel(field_frame, text=value, font=("Arial", 14))
        field_value.pack(side="left", padx=5)

    # Frame for TUIO and Gesture Scores (Below GPA)
    score_frame = ctk.CTkFrame(right_frame, corner_radius=10)
    score_frame.grid(row=len(fields), column=0, padx=20, pady=10, sticky="ew")

    tuio_score_label = ctk.CTkLabel(score_frame, text=f"TUIO Score: {student['tuio_score']}", font=("Arial", 14))
    tuio_score_label.pack(pady=5)

    gesture_score_label = ctk.CTkLabel(score_frame, text=f"Gesture Score: {student['gesture_score']}", font=("Arial", 14))
    gesture_score_label.pack(pady=5)

    # Frame for TUIO and Gesture Images (Column 1)
    image_frame = ctk.CTkFrame(right_frame, corner_radius=10)
    image_frame.grid(row=0, column=1, padx=10, pady=10, rowspan=len(fields) + 1, sticky="nsew")

    # TUIO Image and Text
    try:
        tuio_image = ctk.CTkImage(Image.open("tuio.png"), size=(100, 100))
        tuio_label = ctk.CTkLabel(image_frame, image=tuio_image, text="")
        tuio_label.image = tuio_image  # Prevent garbage collection
        tuio_label.pack(pady=5, padx=20)

        tuio_text_label = ctk.CTkLabel(image_frame, text="Show the ID 0", font=("Arial", 12))
        tuio_text_label.pack(pady=5, padx=10)
    except Exception as e:
        print(f"Error loading TUIO image: {e}")

    # Gesture Image and Text
    try:
        gesture_image = ctk.CTkImage(Image.open("handgesture.jpg"), size=(100, 100))
        gesture_label = ctk.CTkLabel(image_frame, image=gesture_image, text="") 
        gesture_label.image = gesture_image  # Prevent garbage collection
        gesture_label.pack(pady=5, padx=10)

        gesture_text_label = ctk.CTkLabel(image_frame, text="Gesture Recognition", font=("Arial", 12))
        gesture_text_label.pack(pady=5, padx=10)
    except Exception as e:
        print(f"Error loading gesture image: {e}")

    threading.Thread(target=detect_gesture_and_click, args=(student,)).start()

    # Run the application
    app.mainloop()

if __name__ == "__main__":
    create_student_gui(student_data[0])
