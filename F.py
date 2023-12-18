import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import scipy.ndimage
import numpy as np

root = tk.Tk()

canvas1 = tk.Canvas(root, width=400, height=400, bg='white', relief='raised')
canvas1.pack()

label1 = tk.Label(root, text='Image Processing with PyTorch and SciPy', bg='white')
label1.config(font=('helvetica', 16))
canvas1.create_window(200, 30, window=label1)

image_path = ""

def load_image():
    global image_path
    image_path = filedialog.askopenfilename()
    load = Image.open(image_path)
    render = ImageTk.PhotoImage(load)
    img_label = tk.Label(root, image=render)
    img_label.image = render
    canvas1.create_window(200, 150, window=img_label)

load_button = tk.Button(text="Load Image", command=load_image, bg='blue', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(200, 100, window=load_button)

def apply_filter():
    global image_path
    if image_path:
        filter_type = simpledialog.askstring("Filter Type", "Enter filter type (e.g., blur, edge_detect):").lower()

        if filter_type in ["blur", "edge_detect"]:
          
            image = Image.open(image_path)

          
            transform = transforms.ToTensor()
            tensor_image = transform(image).unsqueeze(0)

            
            if filter_type == "blur":
                tensor_image = torch.nn.functional.avg_pool2d(tensor_image, kernel_size=3, padding=1)
            elif filter_type == "edge_detect":
                sobel_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
                tensor_image = torch.nn.functional.conv2d(tensor_image, sobel_filter, padding=1)

        
            tensor_image = tensor_image.squeeze(0)
            pil_image = transforms.ToPILImage()(tensor_image)

            
            save_path = filedialog.asksaveasfilename(defaultextension='.png')
            pil_image.save(save_path)
            messagebox.showinfo("Image Processed", f"The {filter_type} filter has been applied and saved at:\n{save_path}")
        else:
            messagebox.showwarning("Invalid Filter", "Please enter a valid filter type (blur or edge_detect).")
    else:
        messagebox.showwarning("No Image Loaded", "Please load an image before applying filters.")

filter_button = tk.Button(text='Apply Filter', command=apply_filter, bg='blue', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(200, 350, window=filter_button)

root.mainloop()
