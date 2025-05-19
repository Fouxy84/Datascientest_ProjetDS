# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:53:49 2025

@author: coach
"""

# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import clip
import torch_directml

# Device
device = torch_directml.device()
print(f"Device utilisé : {device}")

# Charger modèle CLIP image uniquement
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device, jit=False)

# Charger tokenizer + modèle BERT texte
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).float().to(device)
bert_model.eval()  # on gèle le modèle texte (optionnel)

# Transform image (taille 224x224 pour CLIP)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

# Dataset personnalisé
class MultimodalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['nom']
        text = str(self.data.iloc[idx]['texte_complet_clean'])
        label = int(self.data.iloc[idx]['label'])

        image = Image.open(img_path).convert("RGB")
        image = image_transforms(image)

        return image, text, label

# Modèle multimodal
class MultimodalModel(nn.Module):
    def __init__(self, clip_model, bert_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.bert_model = bert_model
        # Dimensions : 
        # clip visual output dim (ViT-B/32) = 512
        # bert output dim = 768 (bert-base)
        self.fc = nn.Linear(512 + 768, num_classes)

    def forward(self, images, texts):
        # Image features CLIP (sans grad)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)

        # Tokenize textes BERT avec padding + attention mask
        encoding = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        # BERT outputs last hidden state
        outputs = self.bert_model(**encoding)
        # On récupère l'embedding du token [CLS] (batch_size x 768)
        text_features = outputs.last_hidden_state[:, 0, :]

        # Concaténation embeddings
        combined = torch.cat((image_features, text_features), dim=1)
        out = self.fc(combined)
        return out

# --- Chargement des données ---
csv_path = "C:/Users/coach/Desktop/datascientest/Projet DATASCIENTEST/projet_DS/data_final_clean.csv"
dataset = MultimodalDataset(csv_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# --- Initialisation modèle ---
num_classes = 14
model = MultimodalModel(model_clip, bert_model, num_classes).to(device)

# --- Optimiseur et loss ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Scheduler (optionnel)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# --- Entraînement simplifié ---
num_epochs = 10
best_val_loss = float("inf")
patience, patience_counter = 3, 0

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
        torch.save(model.state_dict(), "best_multimodal_model_bert_clip.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

print("Entraînement terminé, meilleur modèle sauvegardé sous best_multimodal_model_bert_clip.pt")
