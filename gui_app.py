import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
import os
import json 
from torchvision import transforms 

# Import necessary components from project
from models.traffic_sign_ann import TrafficSignCNN_AE_ANN
from datasets.traffic_sign_dataset import NUM_CLASSES 
from utils.model_utils import load_model

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Classifier")
        self.root.geometry("800x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for GUI app")

        self.model = None
        self.class_names = self.load_class_names() 
        self.load_model()

        # Define standard normalization parameters (must match training/evaluation)
        self.NORM_MEAN = [0.485, 0.456, 0.406]
        self.NORM_STD = [0.229, 0.224, 0.225]

        # Define the transformation for inference (similar to val_test_transform in main.py)
        self.infer_transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),          
            transforms.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD) 
        ])

        self.create_widgets()
        self.setup_drag_and_drop()

    def load_model(self):
        model_path = 'best_traffic_sign_classifier.pth' # Path to your trained model
        if os.path.exists(model_path):
            self.model = load_model(TrafficSignCNN_AE_ANN, model_path, self.device)
            if self.model:
                self.model.eval() # Set model to evaluation mode
                print("Model loaded successfully for inference.")
            else:
                print("Failed to load model.")
                # self.label_prediction.config(text="Error: Model could not be loaded.", fg="red")
        else:
            print(f"Model file not found at {model_path}. Please train the model first.")
            # self.label_prediction.config(text="Error: Model file not found. Train the model first.", fg="red")

    def load_class_names(self):
        # Load class names from a dummy dataset instance to ensure consistency
        # This dataset isn't used for loading images but to get the class mapping
        try:
            # Point to any valid COCO JSON file from your dataset to get class names
            dummy_json_path = 'data_stratified/test/_annotations.coco.json' 
            # Temporarily use CPU for this dummy dataset init, as we only need class mapping
            dummy_dataset = TrafficSignDataset(json_path=dummy_json_path, image_dir='.', device=torch.device('cpu'))
            
            # Map 0-indexed labels back to original category names based on the created mapping
            # This ensures class_names list is ordered correctly by the internal label index
            sorted_category_ids = sorted(dummy_dataset.label_mapping, key=dummy_dataset.label_mapping.get)
            class_names = [dummy_dataset.category_id_to_name[cat_id] for cat_id in sorted_category_ids]
            
            if len(class_names) != NUM_CLASSES:
                print(f"Warning: Mismatch between NUM_CLASSES ({NUM_CLASSES}) and loaded class names ({len(class_names)}).")
            
            return class_names
        except Exception as e:
            print(f"Error loading class names: {e}")
            # Fallback to dummy names if loading fails
            return [f"Class {i}" for i in range(NUM_CLASSES)]


    def create_widgets(self):
        # Frame for image display
        self.image_frame = tk.Frame(self.root, bd=2, relief="groove")
        self.image_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.image_label = tk.Label(self.image_frame, text="Drop Image Here or Click to Select", 
                                    width=70, height=25, bg="lightgray")
        self.image_label.pack(expand=True, fill="both")
        self.image_label.bind("<Button-1>", self.load_image_dialog)

        # Frame for prediction and probabilities
        self.info_frame = tk.Frame(self.root, bd=2, relief="groove")
        self.info_frame.pack(pady=10, padx=10, fill="x")

        self.label_prediction = tk.Label(self.info_frame, text="Prediction: N/A", font=("Arial", 16))
        self.label_prediction.pack(pady=5)

        self.text_probabilities = tk.Text(self.info_frame, height=8, width=50, state=tk.DISABLED, font=("Arial", 12))
        self.text_probabilities.pack(pady=5)

    def setup_drag_and_drop(self):
        self.root.drop_target_register(tk.DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        file_path = event.data
        if file_path:
            # Remove curly braces if present (Tkinter on some OS might add them)
            if file_path.startswith('{') and file_path.endswith('}'):
                file_path = file_path[1:-1]
            self.process_image(file_path)

    def load_image_dialog(self, event=None):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        if not self.model:
            self.label_prediction.config(text="Error: Model not loaded.", fg="red")
            return

        try:
            image = Image.open(image_path).convert('RGB')
            
            # Display the image
            img_display = image.copy()
            img_display.thumbnail((400, 400)) # Resize for display
            photo = ImageTk.PhotoImage(img_display)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo # Keep a reference!

            # Prepare image for model inference
            input_tensor = self.infer_transform(image).unsqueeze(0).to(self.device) # Add batch dimension

            # Perform inference
            self.model.eval() # Ensure model is in evaluation mode
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy().flatten() # Convert to probabilities

            # Get predicted class index
            predicted_index = np.argmax(probabilities)
            
            # Get predicted label name
            predicted_label_name = "Unknown"
            if 0 <= predicted_index < len(self.class_names):
                predicted_label_name = self.class_names[predicted_index]
                self.label_prediction.config(text=f"Predicted: {predicted_label_name} (Index: {predicted_index})", fg="blue")
            else:
                self.label_prediction.config(text=f"Predicted: {predicted_label_name} (Index: {predicted_index}) - Out of bounds", fg="orange")
            
            # Display top probabilities
            sorted_indices = np.argsort(probabilities)[::-1] 
            top_k = 5 
            top_probabilities_text = "Top Probabilities:\n" 
            for i in range(min(top_k, len(self.class_names))): 
                idx = sorted_indices[i]
                top_probabilities_text += f"{self.class_names[idx]}: {probabilities[idx]:.4f}\n"

            self.text_probabilities.config(state=tk.NORMAL)
            self.text_probabilities.delete(1.0, tk.END)
            self.text_probabilities.insert(tk.END, top_probabilities_text)
            self.text_probabilities.config(state=tk.DISABLED)

        except Exception as e:
            self.label_prediction.config(text=f"Error processing image: {e}", fg="red")
            self.image_label.config(image="", text="Error loading image. Drop Image Here or Click to Select")
            self.text_probabilities.config(state=tk.NORMAL)
            self.text_probabilities.delete(1.0, tk.END)
            self.text_probabilities.insert(tk.END, "")
            self.text_probabilities.config(state=tk.DISABLED)


if __name__ == "__main__":
    # To run this, ensure your 'best_traffic_sign_classifier.pth' model
    # is in the same directory or accessible path, and
    # 'data_stratified/test/_annotations.coco.json' exists for class names.
    # You might need to run main.py first to train and save the model.
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()
