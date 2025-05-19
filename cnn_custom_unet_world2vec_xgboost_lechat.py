# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:50:11 2025

@author: coach
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from gensim.models import Word2Vec
import spacy
import xgboost as xgb
import shap
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import f1_score, classification_report

# Download NLTK tokenizer data
nltk.download('punkt')

# Constants
CSV_PATH = "C:/Users/coach/Desktop/datascientest/Projet DATASCIENTEST/projet_DS/data_final_clean.csv"
IMAGE_SIZE = (500, 500)  # Initial size of the images
NORMALIZE_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORMALIZE_STD = (0.26862954, 0.26130258, 0.27577711)
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Image transformations
def get_image_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

# Custom Dataset
class MultimodalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.image_transforms = get_image_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['nom']
        text = str(self.data.iloc[idx]['texte_complet_clean'])
        label = int(self.data.iloc[idx]['label'])

        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        return image, text, label

# Train Word2Vec model
def train_word2vec(texts):
    # Tokenize texts
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]

    # Train Word2Vec model
    model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model

# U-Net Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define U-Net architecture for 500x500 images
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load data
def load_data(csv_path):
    dataset = MultimodalDataset(csv_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# Training loop for U-Net
def train_unet(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}")

# Extract features from U-Net
def extract_features(model, data_loader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, _, label in tqdm(data_loader, desc="Extracting Features"):
            images = images.to(device)
            feature = model.encoder(images)
            features.append(feature.cpu().numpy())
            labels.append(label.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

# Train XGBoost model
def train_xgboost(image_features, text_features, labels):
    # Combine image and text features
    combined_features = np.hstack((image_features, text_features))

    # Train XGBoost model
    model = xgb.XGBClassifier()
    model.fit(combined_features, labels)

    return model

# Evaluate model
def evaluate_model(model, val_loader, word2vec_model):
    # Select a subset of the validation data for evaluation
    images, texts, labels = next(iter(val_loader))
    images = images.to(device)

    # Generate text embeddings using Word2Vec
    text_embeddings = np.array([np.mean([word2vec_model.wv[word] for word in word_tokenize(text.lower()) if word in word2vec_model.wv], axis=0) for text in texts])

    # Extract features using U-Net
    with torch.no_grad():
        image_features = unet_model.encoder(images).cpu().numpy()

    # Combine image and text features
    combined_features = np.hstack((image_features, text_embeddings))

    # Predict
    predictions = model.predict(combined_features)

    # Calculate F1 score
    f1 = f1_score(labels, predictions, average='weighted')
    print(f"F1 Score (weighted): {f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions))

# Explain model predictions using SHAP
def explain_with_shap(model, val_loader, word2vec_model):
    # Select a subset of the validation data for explanation
    images, texts, labels = next(iter(val_loader))
    images = images.to(device)

    # Generate text embeddings using Word2Vec
    text_embeddings = np.array([np.mean([word2vec_model.wv[word] for word in word_tokenize(text.lower()) if word in word2vec_model.wv], axis=0) for text in texts])

    # Extract features using U-Net
    with torch.no_grad():
        image_features = unet_model.encoder(images).cpu().numpy()

    # Combine image and text features
    combined_features = np.hstack((image_features, text_embeddings))

    # Define a function to get model predictions
    def predict(features):
        return xgboost_model.predict_proba(features)

    # Create a SHAP explainer
    explainer = shap.KernelExplainer(predict, combined_features)

    # Compute SHAP values
    shap_values = explainer.shap_values(combined_features)

    # Plot SHAP values
    shap.summary_plot(shap_values, combined_features)

# Main function
def main():
    # Load data
    train_loader, val_loader = load_data(CSV_PATH)

    # Extract texts for Word2Vec training
    texts = [str(item['texte_complet_clean']) for _, item in pd.read_csv(CSV_PATH).iterrows()]

    # Train Word2Vec model
    word2vec_model = train_word2vec(texts)

    # Initialize U-Net model
    unet_model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)

    # Train U-Net model
    train_unet(unet_model, train_loader, criterion, optimizer, NUM_EPOCHS)

    # Extract features from U-Net
    image_features, labels = extract_features(unet_model, train_loader)

    # Generate text features using Word2Vec
    text_features = np.array([np.mean([word2vec_model.wv[word] for word in word_tokenize(text.lower()) if word in word2vec_model.wv], axis=0) for text in texts[:len(labels)]])

    # Train XGBoost model
    xgboost_model = train_xgboost(image_features, text_features, labels)

    # Evaluate model
    evaluate_model(xgboost_model, val_loader, word2vec_model)

    # Explain model predictions using SHAP
    explain_with_shap(xgboost_model, val_loader, word2vec_model)

if __name__ == "__main__":
    main()
