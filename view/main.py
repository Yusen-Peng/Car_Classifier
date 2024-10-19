from random import randint
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
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
    
def process_image():
    b.config(text = randint(1,10))


root = tk.Tk()
root.title("Car Safety Detection")
a = tk.Label(root, text ="Upload Image for Car Classification") 
b = tk.Label(root, text="")

upload_button = tk.Button(root, text="Select Image", command=upload_image)
image_label = tk.Label(root)
image_label.pack()
process_button = tk.Button(root, text="Process Image", command=process_image)
a.pack() 
upload_button.pack()
b.pack()
process_button.pack()


root.mainloop()