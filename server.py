import customtkinter as ctk
from PIL import Image, ImageTk
from YOLO_TRY import detect_fruits_with_bboxes
import threading
import cv2
import mediapipe as mp
from dollarpy import Recognizer, Template, Point

# Global variables
image_paths = [f"fruit_images/{i}.png" for i in range(1, 5)]
image_info = []
detection_canvas = None  # Canvas to render bounding boxes
fruit_name_label = None
fruit_description_label = None
fruit_benefits_label = None

# Flag to control camera closing and detection flow
camera_closed = False
detection_completed = False  # New flag to track detection completion

def update_gui_with_detections(detections):
    """
    Update the GUI with bounding boxes and detected fruits.

    Args:
        detections (list): List of detected fruits with bounding box details.
    """
    global detection_completed, camera_closed

    fruit_image_map = {
        "apple": "fruit_images/1.png",
        "banana": "fruit_images/2.png",
        "orange": "fruit_images/3.png",
    }
    fruit_info = {
        "apple": {
            "name": "Apple",
            "description": "Apples are sweet and crunchy.",
            "benefits": "Rich in fiber and vitamin C."
        },
        "banana": {
            "name": "Banana",
            "description": "Bananas are soft and creamy.",
            "benefits": "A great source of potassium."
        },
        "orange": {
            "name": "Orange",
            "description": "Oranges are juicy and tangy.",
            "benefits": "High in vitamin C and antioxidants."
        }
    }

    if detection_canvas is None:
        return

    # Clear previous bounding boxes
    detection_canvas.delete("all")

    for detection in detections:
        fruit_name = detection["name"]
        x1, y1, x2, y2 = detection["bbox"]

        # Draw bounding box on the canvas
        detection_canvas.create_rectangle(
            x1, y1, x2, y2, outline="green", width=2
        )
        detection_canvas.create_text(
            x1 + 5, y1 - 10, text=fruit_name, fill="green", anchor="nw", font=("Arial", 12, "bold")
        )

        # Highlight the detected fruit in the gray frame
        if fruit_name in fruit_image_map:
            original_image_path = fruit_image_map[fruit_name]
            original_image = ImageTk.PhotoImage(Image.open(original_image_path).resize((100, 100)))

            for image in image_info:
                if original_image_path in image["image_path"]:
                    image_label = image["label"]

                    # Update the label with the original colored image
                    image_label.configure(image=original_image)
                    image_label.image = original_image  # Prevent garbage collection

                    print(f"Updated GUI with detected fruit: {fruit_name}")

                    # If fruit is detected, stop fruit detection and start pose recognition
                    if not detection_completed:
                        detection_completed = True
                        camera_closed = True
                        print("Fruit detected! Switching to pose recognition.")
                        cap.release()  # Close the camera for fruit detection
                        run_pose_recognition()  # Start gesture/pose recognition

            if fruit_name in fruit_info:
                fruit_details = fruit_info[fruit_name]
                fruit_name_label.configure(text=f"{fruit_details['name']}")
                fruit_description_label.configure(text=f"{fruit_details['description']}")
                fruit_benefits_label.configure(text=f"{fruit_details['benefits']}")

def create_gesture():
    global detection_canvas, fruit_benefits_label, fruit_description_label, fruit_name_label

    root = ctk.CTk()
    root.geometry("800x450+400+250")
    root.title("Fruit Detection GUI")

    mainframe = ctk.CTkFrame(root, width=800, height=450, corner_radius=10)
    mainframe.pack(pady=20)
    mainframe.pack_propagate(False)

    # Frame for gray images
    gray_image_frame = ctk.CTkFrame(mainframe, width=700, height=100, corner_radius=10)
    gray_image_frame.place(x=50, y=20)  # Place near the top center

    start_x = 50
    x_offset = 150

    for col_idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("L").resize((100, 100))
        image_tk = ImageTk.PhotoImage(image)

        x_pos = start_x + (col_idx * x_offset)

        image_label = ctk.CTkLabel(gray_image_frame, image=image_tk, text="")
        image_label.image = image_tk  # Prevent garbage collection
        image_label.place(x=x_pos, y=0)

        image_info.append({
            "id": col_idx + 1,
            "image_path": image_path,
            "label": image_label
        })
    
    # Labels for fruit information
    fruit_name_label = ctk.CTkLabel(
        mainframe, text="", font=("Arial", 24, "bold"), wraplength=250, anchor="w"
    )
    fruit_name_label.place(x=400, y=200)  # Positioned to the right of the detection canvas

    fruit_description_label = ctk.CTkLabel(
        mainframe, text="", font=("Arial", 14), wraplength=250, anchor="w"
    )
    fruit_description_label.place(x=400, y=250)  # Positioned below the fruit name label

    fruit_benefits_label = ctk.CTkLabel(
        mainframe, text="", font=("Arial", 14), wraplength=250, anchor="w"
    )
    fruit_benefits_label.place(x=400, y=300)

    detection_canvas = ctk.CTkCanvas(mainframe, width=300, height=300, bg="black")
    detection_canvas.place(x=50, y=200)  # Position bottom-left of the mainframe

    # Start YOLO detection in a separate thread
    threading.Thread(target=detect_fruits_with_bboxes, args=(update_gui_with_detections,), daemon=True).start()

    root.mainloop()


def run_pose_recognition():
    """
    Run the pose recognition loop. Detects and processes hand gestures in real time using MediaPipe and a recognizer.
    """
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)  # Open webcam
    framecnt = 0
    Allpoints = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Preprocess the frame
            frame = cv2.resize(frame, (480, 320))
            frame = cv2.flip(frame, 1)
            framecnt += 1

            # Convert frame to RGB
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(RGB)

            if results.pose_landmarks:
                # Get right wrist coordinates
                image_height, image_width, _ = frame.shape
                x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width)
                y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height)
                Allpoints.append(Point(x, y, 1))

                # Get left wrist coordinates
                x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width)
                y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height)
                Allpoints.append(Point(x, y, 1))

                # Recognize gestures every 15 frames
                if framecnt % 15 == 0:
                    framecnt = 0
                    result = recognizer.recognize(Allpoints)
                    print(result)  # Print the recognized gesture
                    Allpoints.clear()

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame
            cv2.imshow('Output', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    create_gesture()
