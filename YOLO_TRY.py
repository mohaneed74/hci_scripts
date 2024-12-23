from ultralytics import YOLO
import cv2
import logging
import time
# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

start_time = None
def detect_fruits_with_bboxes(callback):
    """
    Detect fruits using YOLO and trigger a callback with detection details.

    Args:
        callback (function): Function to call with detection details (fruit name and bounding box).
    """
    model = YOLO("yolov8n.pt")

    # Define fruit classes
    fruit_classes = ["apple", "banana", "orange", "watermelon"]

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip for mirror view

        # Run YOLO inference
        results = model.predict(source=frame, save=False, show=False, conf=0.5)

        # Process detections
        detections = []  # List to hold detection details
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                class_name = model.names[int(cls)]
                if class_name in fruit_classes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        "name": class_name,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf.item()
                    })

                    # Draw bounding box and label on the frame
                    label = f"{class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Send detections to the callback
        if detections:
            callback(detections)
            start_time = time.time()
            break
        # Display the frame with detections
        cv2.imshow("Fruit Detection", frame)

        # Exit on pressing the 'ESC' key
        key = cv2.waitKey(1)
        if key == 27:
            print("Exiting...")
            break

    # Release resources
    if time.time() - start_time > 3:
        cap.release()
        cv2.destroyWindow("Fruit Detection")
