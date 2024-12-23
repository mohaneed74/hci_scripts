import threading
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImagePath
import socket
from Students_data import read_highschool_students_from_csv, update_student
from eye_tracker import EyeTracker

count = 0  # Global question index
students_data = read_highschool_students_from_csv('students data.csv')
selected_button_index = None
current_quiz = None
answers_buttons = []  # Declare globally for button updates
lock=threading.Lock()
# Quiz questions
questions = [
    {"question": "What is the capital of France?", "options": ["Paris", "London", "Berlin", "Rome"], "answer": "Paris"},
    {"question": "Which planet is known as the Red Planet?", "options": ["Earth", "Mars", "Jupiter", "Venus"], "answer": "Mars"},
    {"question": "What is the largest ocean on Earth?", "options": ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"], "answer": "Pacific Ocean"},
    {"question": "Which organ pumps blood in the human body?", "options": ["Brain", "Stomach", "Heart", "Liver"], "answer": "Heart"},
    {"question": "What is the freezing point of water?", "options": ["0°C", "10°C", "50°C", "-10°C"], "answer": "0°C"},
    {"question": "Solve: What is the derivative of x^2?", "options": ["2x", "x^2", "x", "2"], "answer": "2x"},
    {"question": "Which programming language is known for web development?", "options": ["Python", "Java", "JavaScript", "C++"], "answer": "JavaScript"},
    {"question": "Who proposed the theory of relativity?", "options": ["Isaac Newton", "Albert Einstein", "Galileo Galilei", "Nikola Tesla"], "answer": "Albert Einstein"},
    {"question": "What is the chemical symbol for water?", "options": ["H2O", "O2", "CO2", "HO2"], "answer": "H2O"},
    {"question": "What does HTTP stand for?", "options": ["Hyper Text Transfer Protocol", "High Transfer Text Protocol", "Hyperlink Text Transfer Program", "High Traffic Transfer Protocol"], "answer": "Hyper Text Transfer Protocol"},
    {"question": "Which of these is a renewable energy source?", "options": ["Solar", "Coal", "Gas", "Oil"], "answer": "Solar"},
    {"question": "What is the square root of 64?", "options": ["8", "6", "7", "9"], "answer": "8"},
    {"question": "What does HTML stand for?", "options": ["Hyper Text Markup Language", "Home Tool Markup Language", "Hyperlinks Text Markup Language", "Hyper Text Management Language"], "answer": "Hyper Text Markup Language"},
    {"question": "Which gas do plants use for photosynthesis?", "options": ["Oxygen", "Carbon Dioxide", "Nitrogen", "Hydrogen"], "answer": "Carbon Dioxide"}
]

def show_message(quiz, text):
    """Displays a temporary message box for 2 seconds."""
    message_box = ctk.CTkLabel(quiz, text=text, font=("Arial", 14), fg_color="gray", text_color="white", corner_radius=10)
    message_box.place(relx=0.5, rely=0.8, anchor="center")
    quiz.after(2000, message_box.destroy)

def button_clicked(answer, correct_answer, button, quiz, qlabel, answers_buttons, student, score_label):
    """Handles button clicks and moves to the next question after a short delay."""
    if answer == correct_answer:
        button.configure(fg_color="green")
        show_message(quiz, "Correct!")
        student['tuio_score'] += 10
    else:
        button.configure(fg_color="red")
        show_message(quiz, "Not Correct!")
        student['tuio_score'] -= 5

    score_label.configure(text=f"{student['name']} : {student['tuio_score']}")
    quiz.after(2000, lambda: NextQ(quiz, qlabel, answers_buttons, student, score_label))

def NextQ(quiz, qlabel, answers_buttons, student, score_label):
    """Updates the question and answers for the next question."""
    global count
    with lock:
        count += 1
        if count < len(questions):
            qlabel.configure(text=questions[count]['question'])
            
            # Reset button states
            for button in answers_buttons:
                button.configure(fg_color="gray", text="")
            
            # Update buttons with new options
            for i, button in enumerate(answers_buttons):
                button.configure(
                    text=questions[count]['options'][i],
                    fg_color="gray",
                    command=lambda btn=button, option=questions[count]['options'][i]: button_clicked(option, questions[count]['answer'], btn, quiz, qlabel, answers_buttons, student, score_label)
                )
        else:
            qlabel.configure(text="Quiz Completed!")
            for button in answers_buttons:
                button.destroy()
            update_student(student=student)

def update_button_color(selected_index):
    """Update the button colors based on the selected answer."""
    global answers_buttons, selected_button_index
    for i, button in enumerate(answers_buttons):
        if i == selected_index:
            button.configure(fg_color="green")  # Highlight selected button
        else:
            button.configure(fg_color="gray")  # Reset other buttons
    selected_button_index = selected_index

def choose_answer_based_on_angle(angle):
    """Choose an answer based on the received marker angle."""
    global selected_button_index
    # Divide 360° into four quadrants
    if 0 <= angle < 30 or 120 <= angle < 150 or 240 <= angle < 270:
        selected_index = 0
    elif 30 <= angle < 60 or 150 <= angle < 180 or 270 <= angle < 300:
        selected_index = 1
    elif 60 <= angle < 90 or 180 <= angle < 210 or 300 <= angle < 330:
        selected_index = 2
    elif 90 <= angle < 120 or 210 <= angle < 240 or 330 <= angle < 360:
        selected_index = 3

    if selected_index != selected_button_index:
        update_button_color(selected_index)

def CreateQuiz(student):
    """Creates the quiz window with questions and options."""
    global count, answers_buttons
    count = 0

    quiz = ctk.CTk()
    quiz.geometry('900x400+400+150')
    quiz.title(f'Question {count + 1}')

    mainframe = ctk.CTkFrame(quiz, width=500, height=400, corner_radius=15)
    mainframe.pack(pady=30)

    score_label = ctk.CTkLabel(mainframe, text=f"{student['name']} : {student['tuio_score']}")
    score_label.place(relx=0.5, rely=0.1, anchor='center')

    question_label = ctk.CTkLabel(mainframe, text=questions[count]['question'], font=("Arial", 16))
    question_label.place(relx=0.5, rely=0.3, anchor='center')

    answers_buttons = []
    for i, option in enumerate(questions[count]['options']):
        button = ctk.CTkButton(
            mainframe, text=option,
            command=lambda opt=option: button_clicked(opt, questions[count]['answer'], button, quiz, question_label, answers_buttons, student, score_label)
        )
        button.place(relx=0.3 + (i % 2) * 0.4, rely=0.5 + (i // 2) * 0.2, anchor='center')
        answers_buttons.append(button)

    quiz.mainloop()

def update_button_color(selected_index):
    """Update the button colors based on the selected answer."""
    global answers_buttons, selected_button_index
    for i, button in enumerate(answers_buttons):
        if i == selected_index:
            button.configure(fg_color="green")  # Highlight selected button
        else:
            button.configure(fg_color="gray")  # Reset other buttons
    selected_button_index = selected_index


# Socket configuration
def start_server_and_quiz(student, client_socket=None):
    """Start the quiz with an existing socket connection or create a new one."""
    global current_quiz

    # If no client socket is provided, create a new connection
    if client_socket is None:
        listensocket = socket.socket()
        Port = 8000
        IP = socket.gethostname()

        try:
            listensocket.bind(('', Port))
            listensocket.listen(1)
            print("Server started at " + IP + " on port " + str(Port))
            print("Waiting for a client to connect...")
            client_socket, address = listensocket.accept()
            print(f"New connection made from {address}")
            eye_tracker = EyeTracker()
            eye_tracker.start()  # Start the thread
        except Exception as e:
            print(f"Error setting up socket: {e}")
            return

    # Start a thread to handle client data
    try:
        threading.Thread(target=handle_client_data, args=(client_socket,), daemon=True).start()
    except Exception as e:
        print(f"Error starting client data thread: {e}")
        return

    # Create the quiz
    try:
        CreateQuiz(student)
    except Exception as e:
        print(f"Error creating quiz: {e}")


def parse_client_data(data):
    """Parse the client data string into a list of data objects."""
    markers=[]
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


def handle_client_data(client_socket):
    """Continuously receive data from the client."""
    global current_quiz, selected_button_index, answers_buttons
    while True:
        data = client_socket.recv(1024).decode('utf-8')
        if data:
            markers = parse_client_data(data)
            if len(markers) == 1:
                for marker in markers:
                    # Choose the answer based on the angle of the marker
                    if marker['id'] == 0:
                        choose_answer_based_on_angle(marker['angle'])
                    elif marker['id'] == 1:
                # If two markers are present, invoke the selected button
                        if selected_button_index is not None:
                            print(f"Invoking button for selected index: {selected_button_index}")
                            answers_buttons[selected_button_index].invoke()
                            selected_button_index=None


# To start the quiz and server with a student:
if __name__=='__main__':
    start_server_and_quiz(student=students_data[0])
