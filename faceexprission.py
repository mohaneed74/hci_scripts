import cv2
from deepface import DeepFace
import time
from collections import Counter
import csv
import os
import tkinter as tk

# Store detected emotions
emotion_data = []

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame):
    """
    Analyze a single frame to detect emotions using DeepFace.
    """
    try:
        # Convert the frame to grayscale (required for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
        
        # Process the first detected face (you can extend this for multiple faces if needed)
        x, y, w, h = faces[0]
        face_frame = frame[y:y+h, x:x+w]

        # Analyze the face with DeepFace
        analysis = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_data.append(dominant_emotion)

        # Display the result on the frame
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        cv2.putText(frame, "Error in analysis", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def save_emotion_data_to_csv():
    """
    Save the emotion data to a CSV file.
    """
    if not emotion_data:
        print("No emotions detected to save.")
        return

    # Create a folder for saving results if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Define the CSV file path
    csv_file_path = os.path.join("results", "emotion_analysis.csv")

    # Count emotions
    emotion_counter = Counter(emotion_data)

    # Write data to CSV
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Emotion", "Occurrences"])  # Write header
        for emotion, count in emotion_counter.items():
            writer.writerow([emotion, count])

    print(f"Emotion analysis saved to {csv_file_path}")

def show_message(message):
    """
    Display a message in a popup window for 3 seconds.
    """
    popup = tk.Toplevel()
    popup.title("Message")
    label = tk.Label(popup, text=message, font=("Arial", 12))
    label.pack(padx=20, pady=20)
    popup.after(3000, popup.destroy)

def analyze_emotions():
    """
    Analyze and display the summary of emotions detected during the session.
    """
    if emotion_data:
        emotion_counter = Counter(emotion_data)
        print("\nEmotion Analysis Summary:")
        for emotion, count in emotion_counter.items():
            print(f"{emotion}: {count} occurrences")
        print("\nMost frequent emotion:", emotion_counter.most_common(1)[0][0])
        show_message(f"Most frequent emotion: {emotion_counter.most_common(1)[0][0]}")
        save_emotion_data_to_csv()  # Save the data to a CSV file
    else:
        print("No emotions detected.")

def start_emotion_analysis():
    """
    Start emotion analysis using webcam feed in a loop.
    """
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam for emotion analysis.")
        return

    try:
        print("Tracking emotions... Press 'Esc' to stop.")
        start_time = time.time()
        interval = 3  # Time interval between analyses

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting emotion analysis.")
                break

            if time.time() - start_time >= interval:
                process_frame(frame)  # Analyze the frame
                start_time = time.time()

            # Display the frame
            cv2.imshow("Emotion Analysis", frame)

            # Exit if 'Esc' is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyWindow("Emotion Analysis")
        analyze_emotions()  # Show emotion summary and save to CSV
