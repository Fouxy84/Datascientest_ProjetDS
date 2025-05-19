# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:57:14 2025

@author: coach
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
import clip
import torch_directml
import shap
import numpy as np
import re

# Constants
CSV_PATH = "C:/Users/coach/Desktop/datascientest/Projet DATASCIENTEST/projet_DS/data_final_clean.csv"
MODEL_SAVE_PATH = "C:/Users/coach/Desktop/datascientest/Projet DATASCIENTEST/projet_DS/best_multimodal_model_distilbert_clip.pt"
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORMALIZE_STD = (0.26862954, 0.26130258, 0.27577711)
BATCH_SIZE = 2
NUM_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 1e-4

# Device configuration
device = torch_directml.device()
print(f"Device utilisé : {device}")

# Load CLIP model for images
def load_clip_model():
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device, jit=False)
    return model_clip, preprocess_clip

# Load DistilBERT model for text
def load_bert_model():
    bert_model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
    bert_model = DistilBertModel.from_pretrained(bert_model_name).float().to(device)
    bert_model.eval()  # Freeze the text model (optional)
    return tokenizer, bert_model

# Image transformations
def get_image_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

# Custom Dataset
# class MultimodalDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#         self.image_transforms = get_image_transforms()

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path = self.data.iloc[idx]['nom']
#         text = str(self.data.iloc[idx]['texte_complet_clean'])
#         label = int(self.data.iloc[idx]['label'])

#         image = Image.open(img_path).convert("RGB")
#         image = self.image_transforms(image)

#         return image, text, label

def clean_text(text):
    # Supprimer les caractères non valides
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

# Utiliser la fonction de nettoyage lors de la lecture des données
class MultimodalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, encoding='latin1')
        self.image_transforms = get_image_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['nom']
        text = str(self.data.iloc[idx]['texte_complet_clean'])
        text = clean_text(text)  # Nettoyer le texte
        label = int(self.data.iloc[idx]['label'])

        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        # Générer des embeddings de texte en utilisant spaCy et Word2Vec
        doc = nlp(text)
        text_embedding = np.mean([word2vec_model.wv[word.text] for word in doc if word.text in word2vec_model.wv], axis=0)

        return image, text_embedding, label















# Multimodal Model
class MultimodalModel(nn.Module):
    def __init__(self, clip_model, bert_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.bert_model = bert_model
        # Dimensions: 512 for CLIP, 768 for DistilBERT
        self.fc = nn.Linear(512 + 768, num_classes)

    def forward(self, images, texts):
        # Image features CLIP (no grad)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)

        # Tokenize texts BERT with padding + attention mask
        encoding = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        # BERT outputs last hidden state
        outputs = self.bert_model(**encoding)
        # Get the embedding of the [CLS] token (batch_size x 768)
        text_features = outputs.last_hidden_state[:, 0, :]

        # Concatenate embeddings
        combined = torch.cat((image_features, text_features), dim=1)
        out = self.fc(combined)
        return out

# Load data
# def load_data(csv_path):
#     dataset = MultimodalDataset(csv_path)
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     return train_loader, val_loader

def load_data(csv_path):
    dataset = pd.read_csv(csv_path, encoding='latin1')  # Essayez 'latin1' ou 'ISO-8859-1'
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience):
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        y_train_true, y_train_pred = [], []

        for images, texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_train_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_train_true.extend(labels.cpu().numpy())

        # Validation
        model.eval()
        val_loss = 0.0
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for images, texts, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images, texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                y_val_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                y_val_true.extend(labels.cpu().numpy())

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    print("Entraînement terminé, meilleur modèle sauvegardé.")

# Explain model predictions using SHAP
def explain_with_shap(model, val_loader):
    # Select a subset of the validation data for explanation
    images, texts, labels = next(iter(val_loader))
    images = images.to(device)
    labels = labels.to(device)

    # Define a function to get model predictions
    def predict(images, texts):
        with torch.no_grad():
            outputs = model(images, texts)
        return outputs.cpu().numpy()

    # Create a SHAP explainer
    explainer = shap.DeepExplainer(predict, [images, texts])

    # Compute SHAP values
    shap_values = explainer.shap_values([images, texts])

    # Plot SHAP values
    shap.summary_plot(shap_values, [images, texts], feature_names=["Image Features", "Text Features"])

# Main function
def main():
    # Load models
    model_clip, _ = load_clip_model()
    global tokenizer, bert_model
    tokenizer, bert_model = load_bert_model()

    # Load data
    train_loader, val_loader = load_data(CSV_PATH)

    # Initialize model
    num_classes = 14
    model = MultimodalModel(model_clip, bert_model, num_classes).to(device)

    # Load the best model weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Explain model predictions using SHAP
    explain_with_shap(model, val_loader)

if __name__ == "__main__":
    main()
