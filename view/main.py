import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
# Created by Bennett Godinho-Nelson initial GUI
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.gif")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((200, 200)) 
        photo = ImageTk.PhotoImage(image)

        image_label = tk.Label(root, image=photo)
        image_label.image = photo  
        image_label.pack()

root = tk.Tk()
root.title("Car Safety Detection")
a = tk.Label(root, text ="Upload Image for Car Classification") 

a.pack() 
upload_button = tk.Button(root, text="Select Image", command=upload_image)
upload_button.pack()

root.mainloop()