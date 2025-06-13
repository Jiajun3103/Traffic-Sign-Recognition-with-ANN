import torch
from torch.utils.data import DataLoader
from models.traffic_sign_ann import TrafficSignCNN_AE_ANN
from datasets.traffic_sign_dataset import TrafficSignDataset, NUM_CLASSES 
from utils.training import train_one_epoch
from utils.evaluation import evaluate_model, get_predictions_and_labels, calculate_topk_accuracy, calculate_f1_score, plot_confusion_matrix, plot_roc_curve
from utils.model_utils import save_model, load_model
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision import transforms 

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
# Define paths to your dataset directories and annotation files
train_json_path = 'data_stratified/train/_annotations.coco.json'
train_image_dir = 'data_stratified/train'
valid_json_path = 'data_stratified/valid/_annotations.coco.json'
valid_image_dir = 'data_stratified/valid'
test_json_path = 'data_stratified/test/_annotations.coco.json'
test_image_dir = 'data_stratified/test'

# Define transformations for training, validation, and testing
# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation and Test transforms (no augmentation, only resize, to tensor, normalize)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = TrafficSignDataset(json_path=train_json_path, image_dir=train_image_dir, device=device, transform=train_transform)
valid_dataset = TrafficSignDataset(json_path=valid_json_path, image_dir=valid_image_dir, device=device, transform=val_test_transform)
test_dataset = TrafficSignDataset(json_path=test_json_path, image_dir=test_image_dir, device=device, transform=val_test_transform)

# Create DataLoaders
batch_size = 32 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for Windows compatibility
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model, Criterion, Optimizer
model = TrafficSignCNN_AE_ANN(device=device).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Added weight_decay for L2 regularization
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Learning rate scheduler

# Training loop
num_epochs = 5
best_val_accuracy = 0.0
model_save_path = 'best_traffic_sign_classifier.pth'

print("Starting training...")
for epoch in range(num_epochs):
    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch_num=epoch)
    
    # Validate
    val_loss, val_accuracy = evaluate_model(model, valid_loader, criterion, device)
    
    # Step the scheduler
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        save_model(model, model_save_path)
        print(f"  New best model saved with accuracy: {best_val_accuracy:.2f}%")

print("Training finished.")

# Load the best model for final evaluation
if os.path.exists(model_save_path):
    model = load_model(TrafficSignCNN_AE_ANN, model_save_path, device)
    if model is None:
        print("Failed to load the model. Exiting.")
        exit()
else:
    print(f"No model found at {model_save_path}. Cannot perform evaluation.")
    exit()

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Get predictions and labels for detailed metrics
all_predictions, all_labels, all_scores = get_predictions_and_labels(model, test_loader, device)

# Calculate and print Top-3 Accuracy
top3_accuracy = calculate_topk_accuracy(torch.tensor(all_scores), torch.tensor(all_labels), k=3)
print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")

# Calculate and print F1-score
f1 = calculate_f1_score(all_predictions, all_labels)
print(f"F1-Score (weighted): {f1:.4f}")

# Get class names from the dataset for plotting
# The label_mapping is {category_id: 0-indexed_label}
# We need to reverse it to {0-indexed_label: category_name}
class_names = [name for id, name in sorted(train_dataset.category_id_to_name.items(), key=lambda item: train_dataset.label_mapping[item[0]])]

# Plot Confusion Matrix
print("\nPlotting Confusion Matrix...")
plot_confusion_matrix(all_predictions, all_labels, class_names)

# Plot ROC Curve
print("\nPlotting ROC Curve...")
plot_roc_curve(all_scores, all_labels, NUM_CLASSES, class_names)

# Display some example predictions
print("\nDisplaying example predictions from test set...")
model.eval() # Set model to evaluation mode
dataiter = iter(test_loader)
try:
    images, labels = next(dataiter)
except StopIteration:
    print("Test loader is empty or iterated through. Cannot display examples.")
    exit() 

images = images.to(device)

outputs = model(images)

# For multi-class classification, get the index of the highest probability
_, predicted_indices = torch.max(outputs, 1) 

fig = plt.figure(figsize=(12, 10)) 
# Display up to 8 images
display_count = min(8, images.shape[0]) 

for i in range(display_count): 
    ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[]) 
    
    # Denormalize image for display
    img = images[i].cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean 
    img = np.clip(img, 0, 1) 

    ax.imshow(img)
    
    true_label_name = class_names[labels[i].item()]
    predicted_label_name = class_names[predicted_indices[i].item()]
    
    ax.set_title(f"True: {true_label_name}\nPred: {predicted_label_name}", 
                 color=("green" if predicted_indices[i] == labels[i] else "red"))

plt.tight_layout()
plt.show()
