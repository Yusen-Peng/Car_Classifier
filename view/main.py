from random import randint
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import font
from gui_utilities import capture_image

globalFont = 'Segoe UI Semibold'
root = tk.Tk()
root.title("Car Safety Detection")

# Set window background to dark grey
root.configure(bg='#2e2e2e',width='500px',height='500px')
root.geometry("500x500")
style = ttk.Style(root)

# Created by Bennett Godinho-Nelson initial GUI
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.gif")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((200, 200))  # Resize the image
        photo = ImageTk.PhotoImage(image)

        # Update the existing image_label with the new image
        image_label.config(image=photo)
        image_label.image = photo
# Arlen Feng - Added take_image 
def take_image(): 
    capture_image()
    file_path = 'captured_image.jpg'
    if file_path:
        image = Image.open(file_path)
        image = image.resize((200, 200))  # Resize the image
        photo = ImageTk.PhotoImage(image)

        # Update the existing image_label with the new image
        image_label.config(image=photo)
        image_label.image = photo

def process_image():
    if(image_label.cget('image') != ''):
        b.config(text=randint(1,10))
    else:
        b.config(text='Please enter image.')

def set_style():
    style.theme_use('clam')
    
    # Modern dark styling for TButton
    style.configure('TButton', font=(globalFont, 12), foreground='white', background='#2e2e2e', padding=6)
    style.map('TButton', background=[('active', '#555555')])

    # Modern dark styling for TLabel
    style.configure('TLabel', font=(globalFont, 14), foreground='white', background='#2e2e2e')

    # Modern dark styling for TEntry
    style.configure('TEntry', foreground='white', background='#2e2e2e', fieldbackground='#2e2e2e', padding=5)

# Call set_style() to apply the styles
set_style()
root.attributes('-alpha', 0.95)
a = ttk.Label(root, text="Upload Image for Car Classification", style="TLabel")
b = ttk.Label(root, text="", style="TLabel")

capture_button = ttk.Button(root, text="Capture Image", command=take_image, style="TButton")

upload_button = ttk.Button(root, text="Select Image", command=upload_image, style="TButton")
image_label = tk.Label(root, bg='#2e2e2e')  # Set background to dark grey
process_button = ttk.Button(root, text="Process Image", command=process_image, style="TButton")

# Pack widgets
a.pack(pady=10) 
upload_button.pack(pady=10)
image_label.pack(pady=10)
b.pack(pady=10)
process_button.pack(pady=10)
capture_button.pack(pady=10)

# Start the main loop
root.mainloop()
