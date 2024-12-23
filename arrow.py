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

# Flag to control camera closing
camera_closed = False
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
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
    from dollarpy import Recognizer, Template, Point
    RotateRight = Template('RotateRight', [
    Point(158,366, 1),
    Point(347,362, 1),
    Point(158,372, 1),
    Point(344,367, 1),
    Point(159,373, 1),
    Point(342,368, 1),
    Point(159,374, 1),
    Point(342,369, 1),
    Point(159,374, 1),
    Point(342,370, 1),
    Point(159,373, 1),
    Point(340,369, 1),
    Point(159,373, 1),
    Point(340,369, 1),
    Point(159,373, 1),
    Point(341,369, 1),
    Point(158,373, 1),
    Point(335,364, 1),
    Point(158,372, 1),
    Point(314,340, 1),
    Point(191,352, 1),
    Point(297,315, 1),
    Point(187,354, 1),
    Point(293,292, 1),
    Point(187,370, 1),
    Point(297,260, 1),
    Point(183,377, 1),
    Point(299,232, 1),
    Point(179,380, 1),
    Point(297,205, 1),
    Point(179,381, 1),
    Point(295,189, 1),
    Point(168,385, 1),
    Point(292,178, 1),
    Point(165,383, 1),
    Point(288,168, 1),
    Point(163,382, 1),
    Point(286,163, 1),
    Point(163,382, 1),
    Point(284,165, 1),
    Point(164,382, 1),
    Point(285,168, 1),
    Point(164,382, 1),
    Point(284,168, 1),
    Point(164,383, 1),
    Point(285,172, 1),
    Point(163,383, 1),
    Point(285,176, 1),
    Point(167,382, 1),
    Point(279,179, 1),
    Point(169,382, 1),
    Point(262,180, 1),
    Point(166,381, 1),
    Point(248,174, 1),
    Point(162,382, 1),
    Point(240,166, 1),
    Point(157,380, 1),
    Point(232,154, 1),
    Point(155,372, 1),
    Point(226,140, 1),
    Point(154,368, 1),
    Point(223,125, 1),
    Point(149,369, 1),
    Point(222,111, 1),
    Point(146,370, 1),
    Point(229,102, 1),
    Point(148,371, 1),
    Point(230,86, 1),
    Point(149,372, 1),
    Point(258,95, 1),
    Point(151,376, 1),
    Point(273,90, 1),
    Point(155,381, 1),
    Point(283,93, 1),
    Point(158,385, 1),
    Point(296,99, 1),
    Point(159,386, 1),
    Point(304,103, 1),
    Point(159,386, 1),
    Point(309,125, 1),
    Point(159,385, 1),
    Point(309,143, 1),
    Point(160,385, 1),
    Point(304,157, 1),
    Point(158,384, 1),
    Point(300,168, 1),
    Point(158,388, 1),
    Point(297,184, 1),
    Point(171,393, 1),
    Point(296,196, 1),
    Point(172,393, 1),
    Point(289,198, 1),
    Point(171,392, 1),
    Point(278,198, 1),
    Point(171,392, 1),
    Point(271,195, 1),
    Point(168,391, 1),
    Point(269,195, 1),
    Point(167,391, 1),
    Point(268,194, 1),
    Point(167,391, 1),
    Point(267,193, 1),
    Point(171,392, 1),
    Point(266,195, 1),
    Point(171,392, 1),
    Point(266,198, 1),
    Point(170,391, 1),
    Point(266,202, 1),
    Point(172,388, 1),
    Point(268,222, 1),
    Point(172,385, 1),
    Point(270,253, 1),
    Point(198,346, 1),
    Point(274,281, 1),
    Point(193,358, 1),
    Point(294,325, 1),
    Point(191,363, 1),
    Point(307,346, 1),
    Point(170,367, 1),
    Point(323,360, 1),
    Point(164,373, 1),
    Point(339,364, 1),
    Point(164,374, 1),
    Point(347,361, 1),
    Point(162,377, 1),
    Point(342,366, 1),
    Point(163,378, 1),
    Point(343,365, 1),
    Point(163,379, 1),
    Point(347,367, 1),
    ])
    RotateLeft = Template('RotateLeft', [
    Point(163,378, 1),
    Point(346,382, 1),
    Point(163,375, 1),
    Point(352,381, 1),
    Point(163,375, 1),
    Point(351,387, 1),
    Point(163,375, 1),
    Point(347,391, 1),
    Point(175,370, 1),
    Point(327,380, 1),
    Point(180,361, 1),
    Point(317,377, 1),
    Point(206,309, 1),
    Point(313,356, 1),
    Point(216,285, 1),
    Point(345,371, 1),
    Point(211,272, 1),
    Point(350,375, 1),
    Point(209,248, 1),
    Point(356,383, 1),
    Point(207,217, 1),
    Point(358,385, 1),
    Point(208,199, 1),
    Point(356,388, 1),
    Point(211,184, 1),
    Point(355,385, 1),
    Point(212,171, 1),
    Point(354,380, 1),
    Point(212,166, 1),
    Point(352,376, 1),
    Point(213,161, 1),
    Point(351,373, 1),
    Point(214,160, 1),
    Point(350,371, 1),
    Point(213,163, 1),
    Point(351,371, 1),
    Point(214,166, 1),
    Point(354,372, 1),
    Point(217,165, 1),
    Point(358,377, 1),
    Point(222,167, 1),
    Point(357,378, 1),
    Point(230,170, 1),
    Point(358,378, 1),
    Point(236,172, 1),
    Point(358,378, 1),
    Point(245,170, 1),
    Point(359,376, 1),
    Point(252,160, 1),
    Point(362,376, 1),
    Point(261,148, 1),
    Point(361,371, 1),
    Point(262,136, 1),
    Point(360,369, 1),
    Point(256,131, 1),
    Point(360,366, 1),
    Point(254,116, 1),
    Point(360,364, 1),
    Point(252,111, 1),
    Point(358,362, 1),
    Point(248,104, 1),
    Point(355,360, 1),
    Point(239,99, 1),
    Point(353,360, 1),
    Point(231,101, 1),
    Point(353,363, 1),
    Point(217,102, 1),
    Point(352,365, 1),
    Point(210,105, 1),
    Point(350,366, 1),
    Point(204,108, 1),
    Point(350,367, 1),
    Point(201,116, 1),
    Point(350,367, 1),
    Point(199,132, 1),
    Point(348,365, 1),
    Point(199,148, 1),
    Point(349,365, 1),
    Point(200,161, 1),
    Point(349,374, 1),
    Point(202,170, 1),
    Point(350,376, 1),
    Point(210,175, 1),
    Point(354,378, 1),
    Point(217,180, 1),
    Point(360,387, 1),
    Point(219,190, 1),
    Point(361,387, 1),
    Point(224,193, 1),
    Point(360,395, 1),
    Point(231,190, 1),
    Point(359,396, 1),
    Point(239,187, 1),
    Point(359,394, 1),
    Point(242,184, 1),
    Point(359,393, 1),
    Point(245,182, 1),
    Point(360,392, 1),
    Point(246,182, 1),
    Point(360,391, 1),
    Point(247,181, 1),
    Point(360,390, 1),
    Point(246,181, 1),
    Point(360,390, 1),
    Point(246,180, 1),
    Point(362,392, 1),
    Point(245,179, 1),
    Point(362,392, 1),
    Point(244,185, 1),
    Point(362,392, 1),
    Point(241,199, 1),
    Point(362,388, 1),
    Point(240,225, 1),
    Point(362,381, 1),
    Point(229,259, 1),
    Point(356,361, 1),
    Point(229,285, 1),
    Point(338,355, 1),
    Point(225,313, 1),
    Point(322,355, 1),
    Point(175,364, 1),
    Point(329,358, 1),
    Point(168,374, 1),
    Point(332,369, 1),
    Point(168,374, 1),
    Point(334,373, 1),
    Point(170,371, 1),
    Point(342,374, 1),
    Point(168,372, 1),
    Point(342,377, 1),
    Point(166,378, 1),
    Point(343,379, 1),
    ])
    recognizer = Recognizer([RotateLeft,RotateRight])
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
