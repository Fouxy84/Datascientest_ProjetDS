# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:59:02 2025

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
from transformers import BertTokenizer, BertModel
import clip
import torch_directml
import shap
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Configuration appareil
device = torch_directml.device()
print(f"Appareil utilisé : {device}")

# Chargement modèle CLIP
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device, jit=False)

# Chargement tokenizer + modèle BERT
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).to(device)
bert_model.eval()

# Normalisation CLIP
image_transforms = preprocess_clip

# Dataset personnalisé avec embeddings précalculés
class MultimodalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['nom']
        text = str(row['texte_complet_clean'])
        label = int(row['label'])

        # Image
        image = Image.open(img_path).convert("RGB")
        image_tensor = image_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_emb = model_clip.encode_image(image_tensor).squeeze(0).float().cpu()

        # Texte
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            bert_out = bert_model(**encoding)
            text_emb = bert_out.last_hidden_state[:, 0, :].squeeze(0).float().cpu()

        return image_emb, text_emb, label

# Modèle simple sur embeddings
class MultimodalClassifier(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, num_classes=14):
        super().__init__()
        self.fc = nn.Linear(image_dim + text_dim, num_classes)

    def forward(self, image_emb, text_emb):
        x = torch.cat((image_emb, text_emb), dim=1)
        return self.fc(x)

# Chargement des données
csv_path = "C:/Users/coach/Desktop/datascientest/Projet DATASCIENTEST/projet_DS/data_final_clean.csv"

## reduire les données pour reduire le temps de calcul, 4000 au lieu de 84916
df = pd.read_csv(csv_path)
print(df['label'].value_counts())
# Stratified sampling pour garder les proportions des classes
splitter = StratifiedShuffleSplit(n_splits=1, test_size=(len(df) - 4000), random_state=42)
for train_index, _ in splitter.split(df, df['label']):
    df_sampled = df.iloc[train_index].reset_index(drop=True)
# Sauvegarde pour utilisation dans le Dataset
print(df_sampled['label'].value_counts())
sampled_csv_path = "C:/Users/coach/Desktop/datascientest/Projet DATASCIENTEST/projet_DS/data_sampled_4000.csv"
df_sampled.to_csv(sampled_csv_path, index=False)
# Utilisation dans le Dataset
dataset = MultimodalDataset(sampled_csv_path)
## reduire les données pour reduire le temps de calcul, 4000 au lieu de 84916

#dataset = MultimodalDataset(csv_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Initialisation du modèle
model = MultimodalClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Entraînement
num_epochs = 5
best_val_loss = float("inf")
patience, patience_counter = 3, 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for image_emb, text_emb, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
        image_emb, text_emb, labels = image_emb.to(device), text_emb.to(device), labels.to(device)
        outputs = model(image_emb, text_emb)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for image_emb, text_emb, labels in val_loader:
            image_emb, text_emb, labels = image_emb.to(device), text_emb.to(device), labels.to(device)
            outputs = model(image_emb, text_emb)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

print("Entraînement terminé.")

# --------------------------------------------------
# SHAP : interprétation du texte
# --------------------------------------------------
# ⚠️ SHAP ne supporte pas DirectML : on fait l'interprétation CPU
bert_model.cpu()
model.cpu()

# Choix d’un sous-échantillon pour l’explication
sample_texts = [dataset[i][1] for i in range(100)]  # on récupère les embeddings texte

def predict_text(text_emb_batch):
    image_dummy = torch.zeros((len(text_emb_batch), 512))  # vecteurs image nuls
    logits = model(image_dummy, torch.tensor(text_emb_batch, dtype=torch.float32))
    return logits.detach().numpy()

explainer = shap.Explainer(predict_text, np.stack(sample_texts))
shap_values = explainer(np.stack(sample_texts[:10]))

# Visualisation SHAP
shap.plots.bar(shap_values)
