import customtkinter as ctk
import socket
import threading
from PIL import Image, ImageTk
from tkinter import messagebox
import cv2
from Students_data import read_highschool_students_from_csv
from QuizGen import start_server_and_quiz
student_data=read_highschool_students_from_csv('students data.csv')
listOfVideos=[]
# Define global variables
image_paths = [f"fruit_images/{i}.png" for i in range(1, 5)]
image_info = []
moving_images_info = []
marker_info=[]
gray_image_frame = None
fruit_name_label = None
fruit_description_label = None
fruit_benefits_label = None
mainframe = None  # Define mainframe as a global variable
def show_message(root, text):
    """Displays a temporary message box for 2 seconds."""
    message_box = ctk.CTkLabel(root, text=text, font=("Arial", 14), fg_color="gray", text_color="white", corner_radius=10)
    message_box.place(relx=0.5, rely=0.8, anchor="center")
    root.after(2000, message_box.destroy)
# Initialize the fruit info label
fruit_info_label = None
def CreateLearnScreen(root, student):
    global gray_image_frame, mainframe, fruit_info_label, fruit_name_label, fruit_description_label, fruit_benefits_label  # Declare as global
    root.geometry('800x450+400+250')
    root.title(f"Welcome {student['name']}")
    # Create main frame
    mainframe = ctk.CTkFrame(root, width=700, height=450, corner_radius=10)
    mainframe.pack(pady=20)
    mainframe.configure(bg_color="black")
    mainframe.pack_propagate(False)
    # Create the frame for gray images
    gray_image_frame = ctk.CTkFrame(mainframe, width=700, height=(int(450 / 2) - 100), corner_radius=10)
    gray_image_frame.place(x=0, y=0)

    start_x = 100
    start_y = 20
    x_offset = 150  # Distance between each image in the row

    try:
        # Loop over the image paths
        for col_idx, image_path in enumerate(image_paths):
            # Open and process the image (grayscale background)
            original_image = Image.open(image_path)
            grayscale_image = original_image.convert("L")  # Convert to grayscale
            resized_image = grayscale_image.resize((100, 100))  # Resize image

            image_tk = ImageTk.PhotoImage(resized_image)  # Convert to a format tkinter can use

            x_pos = start_x + (col_idx * x_offset)
            y_pos = start_y

            image_label = ctk.CTkLabel(gray_image_frame, image=image_tk, text="")  # Create label for image
            image_label.image = image_tk  # Prevent garbage collection
            image_label.place(x=x_pos, y=y_pos)

            image_info.append({
                'id': int(col_idx + 1),  # Assign an ID based on index (1, 2, 3, 4)
                'image_path': image_path,
                'x': x_pos,
                'y': y_pos,
                'label': image_label,
                'image': image_tk  # Store initial image as well
            })
       # Create a label to display fruit name, description, and benefits
            fruit_name_label = ctk.CTkLabel(mainframe, text="", font=("Arial", 24, "bold"), wraplength=650, anchor="w")
            # Dynamically position it below the gray image frame
            fruit_name_label.place(relx=0.5, rely=0.4, anchor="center")  # Centered below gray image frame

            fruit_description_label = ctk.CTkLabel(mainframe, text="", font=("Arial", 14), wraplength=650, anchor="w")
            # Adjust the position to be below the fruit name label
            fruit_description_label.place(relx=0.5, rely=0.65, anchor="center")  # Below the fruit name label

            fruit_benefits_label = ctk.CTkLabel(mainframe, text="", font=("Arial", 14), wraplength=650, anchor="w")
            # Adjust the position to be below the description label
            fruit_benefits_label.place(relx=0.5, rely=0.7, anchor="center")  # Below the description label

        
    except Exception as e:
                print(f"Error loading or processing image: {e}")



def parse_client_data(data):
    """Parse the client data string into a list of data objects."""
    markers = []
    try:
        if ";" in data:
            for marker_info in data.split(";"):
                marker_info = marker_info.strip()
                if marker_info:
                    marker_id, marker_angle, marker_x, marker_y = marker_info.split(",")
                    markers.append({
                        'id': int(marker_id),
                        'angle': float(marker_angle),
                        'x': float(marker_x) * 800,
                        'y': float(marker_y) * 450,
                        'ishit': False

                    })
        else:
            marker_id, marker_angle, marker_x, marker_y = data.split(",")
            markers.append({
                'id': int(marker_id),
                'angle': float(marker_angle),
                'x': float(marker_x) * 800,
                'y': float(marker_y) * 450,
                'ishit': False
            })
    except ValueError as e:
        print(f"Invalid data format received: {data} - Error: {e}")
    return markers

# The rest of the code remains the same as in your original script
def update_gui_based_on_marker(marker, student):
    fruit_name_label.configure(text="")
    fruit_description_label.configure(text="")
    fruit_benefits_label.configure(text="")
    # Find existing image in moving_images_info
    cat = None
    for image in moving_images_info:
        if marker['id'] == image['id']:
            cat = image
            break

    stick = False
    for id in array:
        if id == marker['id']:
            stick = True

    # If the marker is not in moving_images_info, create the image
    if cat is None:
        image_path = f'fruit_images/{marker["id"]}.png'
        if image_path in image_paths:
            try:
                # Load the image
                new_image = Image.open(image_path)
                resized_image = new_image.resize((100, 100))
                image_tk = ImageTk.PhotoImage(resized_image)

                # Create and place the image
                image_label = ctk.CTkLabel(mainframe, image=image_tk, text="")
                image_label.image = image_tk
                image_label.place(x=marker['x'], y=marker['y'])

                # Store the image information
                cat = {
                    'id': int(marker['id']),
                    'label': image_label,
                    'image': image_tk,
                    'x': marker['x'],
                    'y': marker['y'],
                    'is_there': False,
                    'image_path': image_path
                }
                moving_images_info.append(cat)
            except Exception as e:
                print(f"Error creating image for marker {marker['id']}: {e}")
                return  # Exit if the image creation fails

    # If `cat` is still None, something went wrong; return early
    if cat is None:
        print(f"Error: Unable to find or create image for marker {marker['id']}")
        return

    # Update the position of the existing image
    if not stick:
        cat['label'].place(x=marker['x'], y=marker['y'])
        cat['x'] = marker['x']
        cat['y'] = marker['y']
        print(f"Updated image position: {cat}")
        for gray in image_info:
            if gray['id'] == cat['id']:
                if not cat['is_there']:  # If the image hasn't stuck yet
                    stickiness_range = 20  # Adjustable range for sticking
                    if (gray['x'] - stickiness_range <= cat['x'] + 50 <= gray['x'] + 100 + stickiness_range) and \
                            (gray['y'] - stickiness_range <= cat['y'] + 50 <= gray['y'] + 100 + stickiness_range):
                        cat['is_there'] = True
                        array.append(cat['id'])
                        print(f"sticks id : {array}")
                        cat['x'] = gray['x']
                        cat['y'] = gray['y']
                        print(f'List has been updated !!!!! {cat}')
                        show_message(root,"that's correct")
                        # Once stuck, align the position with the gray image
                        cat['label'].place(x=cat['x'], y=cat['y'])
                        break

# Global variables to store video playback state
current_client_socket = None
current_video = None
current_video_thread = None
video_playing = False  # To track if a video is currently playing

def rotate(marker, student):
    global current_video, current_video_thread, video_playing  # Access global variables
    show_message(root,"rotate right to show information , rotate left for fun fact, rotate 180 to start quiz")
    # Check if the student has gender information
    gender_pronoun = "he" if student['gender'].lower() == 'male' else "she"
    is_child = student.get('age', 0) <= 12  # Assuming 'age' field exists and children are <= 12 years old
    smiley = "😊" if is_child else ""

    fruit_info = {
        1: {"name": "Apple", "description": f"An apple is crunchy and sweet. {gender_pronoun} will enjoy it! {smiley}", "benefits": "Apples help you stay healthy."},
        2: {"name": "Banana", "description": f"A banana is yellow and soft. {gender_pronoun} will love it! {smiley}", "benefits": "Bananas provide potassium and energy."},
        3: {"name": "Orange", "description": f"An orange is juicy and tangy. {gender_pronoun} will enjoy it! {smiley}", "benefits": "Oranges are rich in vitamin C."},
        4: {"name": "Watermelon", "description": f"A watermelon is a large fruit with green skin and red inside. {gender_pronoun} will love eating it! {smiley}", "benefits": "Watermelon helps you stay hydrated."}
    }
    
    for id in array:
        if id == marker['id']:
            if 50 < marker['angle'] < 65:
                if marker['id'] in fruit_info:
                    fruit_name_label.configure(text=fruit_info[marker['id']]['name'])
                    fruit_description_label.configure(text=fruit_info[marker['id']]['description'])
                    fruit_benefits_label.configure(text=fruit_info[marker['id']]['benefits'])
            elif 250 < marker['angle'] < 300:
                # If a new marker is detected and there is an active video, stop it # Wait for the video thread to finish
                # Run the new video in a separate thread 
                  if marker['id'] in fruit_info:
                    # Display a fun fact or trivia about the fruit
                    fun_fact = {
                        1: "Did you know? Apples are grown on apple trees that can live up to 100 years!",
                        2: "Fun Fact: Bananas float in water due to their unique shape!",
                        3: "Orange trivia: The word 'orange' was the first color to be named after a fruit.",
                        4: "Watermelon is 92% water, making it one of the most hydrating fruits!"
                    }
                    fruit_name_label.configure(text=f"Fun Fact about {fruit_info[marker['id']]['name']}:")
                    fruit_description_label.configure(text=fun_fact[marker['id']])
                    fruit_benefits_label.configure(text="Try touching the fruit to interact!")
                    
            if 180 <= marker['angle'] <= 185:
                root.destroy()
                start_server_and_quiz(student, current_client_socket)

array=[]
# Ensure CreateLearnScreen, handle_client_data, start_server, and create_server_gui functions
# are implemented exactly as they were in your original script
def handle_client_data(client_socket, student):
    """Continuously receive data from the client and update the GUI."""
    while True:
        data = client_socket.recv(1024).decode('utf-8')
        if data:
            markers = parse_client_data(data)  # Parse the received data
            # print(f"Received markers: {markers}")
            for marker in markers:
                if marker not in marker_info:
                    marker_info.append(marker) 
                    print(f"marker angle : {marker['angle']}") # Save marker to marker_info if not already there
                    update_gui_based_on_marker(marker, student)
                    rotate(marker,student)
                else:
                    # Update marker if already exists (no need to add again)
                    update_gui_based_on_marker(marker, student)
                    rotate(marker,student)


def start_server(root, student):
    """Start the socket server and wait for a connection."""
    listensocket = socket.socket()
    Port = 8000
    IP = socket.gethostname()
    listensocket.bind(('', Port))
    listensocket.listen(1)
    print(f"Server started at {IP} on port {Port}")
    print("Waiting for a client to connect...")
    client_socket, address = listensocket.accept()
    print(f"New connection made from {address}")
    
    # Start a thread to handle client data
    threading.Thread(target=handle_client_data, args=(client_socket, student), daemon=True).start()
    
    # Update GUI to start the Learn Screen after server starts
    root.after(0, lambda: CreateLearnScreen(root, student))
    
    return client_socket  # Return the client socket

def create_server_gui(student):
    """Create the server GUI and start the server."""
    global root, current_client_socket
    root = ctk.CTk()
    root.geometry("400x200")
    root.title("Server Status")
    
    # Start the server and get the client socket
    current_client_socket = start_server(root, student)
    
    root.mainloop()


if __name__ == '__main__':
    # Example student data
    create_server_gui(student_data[1])


