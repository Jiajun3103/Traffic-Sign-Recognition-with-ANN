import torch
import torch.nn as nn
from datasets.traffic_sign_dataset import NUM_CLASSES

class TrafficSignCNN_AE_ANN(nn.Module):
    """
    Convolutional Neural Network (CNN) with an Autoencoder (AE) component
    and an Artificial Neural Network (ANN) for classification.
    """
    def __init__(self, device):
        super(TrafficSignCNN_AE_ANN, self).__init__()
        self.device = device

        # CNN Feature Extractor
        # Input size: 3x224x224
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32x112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 64x56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 128x28x28
        )

        # Calculate the flattened size after feature extraction
        # This assumes an input size of 224x224 and the max pooling layers
        self.flattened_size = 128 * 28 * 28 

        # Autoencoder (Encoder part)
        self.encoder = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Autoencoder (Decoder part) - Used for reconstruction, not directly in classification
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.flattened_size),
            nn.Sigmoid()  # Sigmoid to output pixel values between 0 and 1
        )

        # Classifier (Artificial Neural Network)
        # It takes the encoded features from the autoencoder
        # NUM_CLASSES should be defined based on your dataset (e.g., 63 for GTSRB) 
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Add dropout for regularization
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128), # Add BatchNorm for better training stability
            nn.LeakyReLU(0.01),  # Using LeakyReLU as an alternative to ReLU
            nn.Dropout(0.5), # Another dropout layer
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        # Pass through the CNN feature extractor
        x = self.feature_extractor(x)
        
        # Flatten the output for the autoencoder and classifier
        x = x.view(-1, self.flattened_size)

        # Pass through the encoder
        encoded = self.encoder(x)
        
        # Pass through the classifier
        classification_output = self.classifier(encoded)
        
        return classification_output
