
import csv
students_data = [
    {
        'id': 'STU001',
        'name': 'John Smith',
        'age': 16,
        'grade': '10th',
        'gender': 'Male',
        'student_id': 'HS001',
        'school': 'Springfield High School',
        'class': '10-A',
        'gpa': 3.8,
        'email': 'john.smith@studentmail.com',
        'photo_path': 'student_images/1.jpg',
        'tuio_score': 0,
        'gesture_score': 0,
        'mac_address': '00:1A:2B:3C:4D:5E'
    },
    {
        'id': 'STU002',
        'name': 'Sarah Johnson',
        'age': 15,
        'grade': '9th',
        'gender': 'Female',
        'student_id': 'HS002',
        'school': 'Springfield High School',
        'class': '9-B',
        'gpa': 3.7,
        'email': 'sarah.johnson@studentmail.com',
        'photo_path': 'student_images/2.jpg',
        'tuio_score': 0,
        'gesture_score': 0,
        'mac_address': '11:2B:3C:4D:5E:6F'
    },
    {
        'id': 'STU003',
        'name': 'Michael Chen',
        'age': 17,
        'grade': '11th',
        'gender': 'Male',
        'student_id': 'HS003',
        'school': 'Riverside High School',
        'class': '11-C',
        'gpa': 3.5,
        'email': 'michael.chen@studentmail.com',
        'photo_path': 'student_images/3.jpg',
        'tuio_score': 0,
        'gesture_score': 0,
        'mac_address': 'F8:20:A9:EA:1A:16'
    },
    {
        'id': 'STU004',
        'name': 'Emily Davis',
        'age': 14,
        'grade': '8th',
        'gender': 'Female',
        'student_id': 'HS004',
        'school': 'Riverside Middle School',
        'class': '8-D',
        'gpa': 4.0,
        'email': 'emily.davis@studentmail.com',
        'photo_path': 'student_images/4.jpg',
        'tuio_score': 0,
        'gesture_score': 0,
        'mac_address': '33:4D:5E:6F:7G:8H'
    },
    {
        'id': 'STU005',
        'name': 'MR. Dexter',
        'age': 17,
        'grade': '10th',
        'gender': 'Male',
        'student_id': 'HS005',
        'school': 'Riverside Middle School',
        'class': '8-D',
        'gpa': 4.0,
        'email': 'Dexter.sha3ban@studentmail.com',
        'photo_path': 'student_images/5.jpg',
        'tuio_score': 0,
        'gesture_score': 0,
        'mac_address': 'F8:AF:05:99:63:18'
    }
]

# Function to write the high school student data to a CSV file
def write_highschool_students_to_csv(file_name, data):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = list(data[0].keys())  # Extract the keys as column headers
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write each student's data
        for student in data:
            writer.writerow(student)


# Write the student data to the CSV
def read_highschool_students_from_csv(file_name):
    students = []
    with open(file_name, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert numeric fields back to the appropriate data types
            row['age'] = int(row['age'])
            row['gpa'] = float(row['gpa'])
            row['tuio_score'] = int(row['tuio_score'])
            row['gesture_score'] = int(row['gesture_score'])
            students.append(row)
    return students

def update_student(student,students_data=students_data):
    for i, existing_student in enumerate(students_data):
        if existing_student['id'] == student['id']:
            students_data[i] = student  # Update the student data
            print(f"Student with ID {student['id']} has been updated.")
            write_highschool_students_to_csv('students data.csv',students_data)
            print("the file updated !!!")
            return
    print(f"Student with ID {student['id']} not found.")