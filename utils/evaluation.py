import torch
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 

def evaluate_model(model, loader, criterion, device):
    """
    Evaluates the model on a given data loader for MULTI-CLASS (SINGLE-LABEL) classification.
    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The data loader for evaluation.
        criterion (torch.nn.Module): The loss function (expected CrossEntropyLoss for multi-class).
        device (torch.device): The device (CPU or GPU) to perform evaluation on.
    Returns:
        tuple: A tuple containing (average validation loss, accuracy).
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device)) 
            val_loss += loss.item()
            
            # For multi-class classification, get the class with the highest score
            # torch.max returns (values, indices)
            _, predicted = torch.max(outputs, 1) 
            
            # Compare with true labels (which are expected to be class indices)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(loader)
    return avg_val_loss, accuracy

def get_predictions_and_labels(model, loader, device):
    """
    Gets all true labels, raw prediction logits, and scores for ROC curve plotting.
    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The data loader (e.g., test_loader).
        device (torch.device): The device (CPU or GPU).
    Returns:
        tuple: (numpy array of true labels, numpy array of raw logits, numpy array of raw logits for ROC)
    """
    model.eval()
    all_labels = []
    all_predictions_logits = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_labels.extend(labels.cpu().numpy())
            all_predictions_logits.extend(outputs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions_logits = np.array(all_predictions_logits)
    all_scores_for_roc = all_predictions_logits
    return all_labels, all_predictions_logits, all_scores_for_roc

def calculate_topk_accuracy(outputs, labels, k=3):
    """
    Calculates the top-k accuracy.
    Args:
        outputs (torch.Tensor): Model outputs (logits).
        labels (torch.Tensor): True labels.
        k (int): The 'k' for top-k accuracy.
    Returns:
        float: Top-k accuracy.
    """
    with torch.no_grad():
        max_k_values, top_k_indices = torch.topk(outputs, k, dim=1)
        labels_expanded = labels.view(-1, 1).expand_as(top_k_indices)
        correct_top_k = (top_k_indices == labels_expanded).any(dim=1).sum().item()
        return correct_top_k / labels.size(0) * 100

def calculate_f1_score(predictions, labels, average_type='weighted'):
    """
    Calculates the F1-score.
    Args:
        predictions (np.array): Predicted labels.
        labels (np.array): True labels.
        average_type (str): Type of averaging for F1-score (e.g., 'weighted', 'macro', 'micro').
    Returns:
        float: F1-score.
    """
    return f1_score(labels, predictions, average=average_type)

def plot_confusion_matrix(predictions, labels, class_names):
    """
    Plots the confusion matrix.
    Args:
        predictions (np.array): Predicted labels.
        labels (np.array): True labels.
        class_names (list): List of class names.
    """
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(len(class_names)+2, len(class_names)+2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(true_labels_indices, predicted_probabilities, num_classes, class_names):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for each class.
    Args:
        true_labels_indices (np.array): True labels (integer indices from 0 to num_classes-1).
                                         Shape: (num_samples,).
        predicted_probabilities (np.array): Predicted probabilities (softmax outputs) for all classes.
                                            Shape: (num_samples, num_classes).
        num_classes (int): Total number of classes.
        class_names (list): List of class names corresponding to indices.
    """
    plt.figure(figsize=(18, 14)) 

    true_labels_one_hot = label_binarize(true_labels_indices, classes=range(num_classes))
    
    warnings_encountered = False
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) 
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from sklearn.exceptions import UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        for i in range(num_classes):
            if np.sum(true_labels_one_hot[:, i]) > 0:
                fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], predicted_probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
            else:
                print(f"Skipping ROC plot for class {class_names[i]} (index {i}) due to no positive true samples in the data.)")
                warnings_encountered = True

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    if warnings_encountered:
        plt.title('ROC Curve (Some classes skipped due to no positive samples)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)  
