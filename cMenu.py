import tkinter as tk
import math

class CircularMenu:
    def __init__(self, root, radius=100, items=None):
        self.root = root
        self.radius = radius
        self.items = items or []
        self.buttons = []
        
        # Create a Canvas to hold the buttons
        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()
        
        # Draw the circular menu
        self.create_menu()
    
    def create_menu(self):
        angle_gap = 360 / len(self.items)  # Calculate the angle between each button
        
        for i, item in enumerate(self.items):
            angle = i * angle_gap
            x = 150 + self.radius * math.cos(math.radians(angle))
            y = 150 + self.radius * math.sin(math.radians(angle))
            
            button = tk.Button(self.root, text=item, command=lambda item=item: self.on_click(item))
            self.buttons.append(button)
            self.canvas.create_window(x, y, window=button)
    
    def on_click(self, item):
        print(f"Clicked on {item}")

# Set up the main window
root = tk.Tk()
root.title("Circular Menu Example")

# List of menu items
items = ["Home", "Settings", "Profile", "Help"]

# Create the CircularMenu
menu = CircularMenu(root, radius=100, items=items)

root.mainloop()
