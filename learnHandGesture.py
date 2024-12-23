import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
from YOLO_TRY import detect_fruits_with_bboxes
import threading
from Testing import run_pose_recognition
from Students_data import read_highschool_students_from_csv
# Global variables
image_paths = [f"fruit_images/{i}.png" for i in range(1, 5)]
image_info = []
detection_canvas = None  # Canvas to render bounding boxes
dollardetect = False
def update_gui_with_detections(detections):
    """
    Update the GUI with bounding boxes and detected fruits.

    Args:
        detections (list): List of detected fruits with bounding box details.
    """
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
                     # Wait for gesture recognition (either RotateLeft or RotateRight)
                    gesture_recognized =run_pose_recognition()
            if fruit_name in fruit_info and gesture_recognized == True:
                    fruit_details = fruit_info[fruit_name]
                    # fruit_name_label.configure(text=f"{fruit_details['name']}")
                    # fruit_description_label.configure(text=f"{fruit_details['description']}")
                    # fruit_benefits_label.configure(text=f"{fruit_details['benefits']}"
                    fruit_name_label.configure(text=f"{fruit_details['name']}")
                    fruit_description_label.configure(text=f"{fruit_details['description']}")
                    fruit_benefits_label.configure(text=f"{fruit_details['benefits']}")
                    threading.Thread(target=detect_fruits_with_bboxes, args=(update_gui_with_detections,), daemon=True).start()


def create_gesture(student):
    global detection_canvas,fruit_benefits_label,fruit_description_label,fruit_name_label

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
    fruit_benefits_label.place(x=400, y=300)  # 
    detection_canvas = ctk.CTkCanvas(mainframe, width=300, height=300, bg="black")
    detection_canvas.place(x=50, y=200)  # Position bottom-left of the mainframe

    # Start YOLO detection in a separate thread
    threading.Thread(target=detect_fruits_with_bboxes, args=(update_gui_with_detections,), daemon=True).start()

    root.mainloop()

if __name__ == '__main__':

    students=read_highschool_students_from_csv("students data.csv")
    create_gesture(students[0])
