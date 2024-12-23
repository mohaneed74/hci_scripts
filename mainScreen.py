import customtkinter as ctk
import asyncio
import threading
from bleak import BleakScanner
from queue import Queue
from Students_data import read_highschool_students_from_csv
from display_studentData import create_student_gui
from Face_Main import Face_recog

# Now import the function from Face_Main
# Global variables
progress_queue = Queue()
device_found = False
foundedSt = None
students = read_highschool_students_from_csv('students data.csv')


async def scan_devices():
    """Async function to scan for Bluetooth devices"""
    global device_found, foundedSt
    print("Scanning for Bluetooth devices...")
    total_scan_time = 10  # Total scanning time in seconds
    scan_interval = 0.5   # Check for devices every 0.5 seconds
    iterations = int(total_scan_time / scan_interval)
    
    for i in range(iterations):
        # Update progress
        progress = i / iterations
        progress_queue.put((progress, f"Scanning for devices... {int(progress * 100)}%"))
        
        # Scan for devices
        devices = await BleakScanner.discover(timeout=scan_interval)
        if devices:
            for st in students:
                for dev in devices:
                    print(f'{dev.name}::::{dev.address}')
                    if dev.address == st['mac_address']:
                        device_found = True
                        foundedSt = st
                        progress_queue.put((1.0, f"Welcome on board! {st['name']}"))
                        # root.destroy()
                        return
        
        await asyncio.sleep(0.1)
    
    if not device_found:
        progress_queue.put((1.0, "Scan complete - Device not found"))


def run_async_scan():
    """Run the async scanning function in a separate thread"""
    asyncio.run(scan_devices())


def update_gui():
    """Check for updates from the scanning thread and update the GUI"""
    if not progress_queue.empty():
        progress, message = progress_queue.get()
        root.after(0, lambda: [progress_bar.set(progress), label.configure(text=message)])
    
    # If a device is found, transition to the student GUI
    if device_found and foundedSt is not None:
        root.after(0, lambda: [root.destroy(),Face_recog(foundedSt)])
    else:
        # Schedule the next update
        root.after(100, update_gui)


def create_gui():
    """Create and setup the GUI window"""
    global root, label, progress_bar
    
    # Create main window
    root = ctk.CTk()
    root.geometry("800x500+400+250")
    root.title("Welcome on board")
    
    # Create label
    label = ctk.CTkLabel(root, text="Starting scan...", font=("Arial", 16))
    label.pack(pady=60)
    
    # Create progress bar
    progress_bar = ctk.CTkProgressBar(root, width=300)
    progress_bar.pack(pady=60)
    progress_bar.set(0)
    
    # Start the scanning thread
    scanner_thread = threading.Thread(target=run_async_scan, daemon=True)
    scanner_thread.start()
    
    # Start checking for updates
    root.after(100, update_gui)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    create_gui()
