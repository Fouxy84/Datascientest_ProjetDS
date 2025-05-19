# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:29:16 2025

@author: coach
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import spacy
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Set device
device = torch.device("cpu")#("cuda" if torch.cuda.is_available() else "cpu")

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Paths
csv_path = "C:/Users/coach/Desktop/datascientest/Projet DATASCIENTEST/projet_DS/data_final_clean.csv"
image_folder = "C:/User/Coach/Desktop/datascientest/Projet DATASCIENTEST/projet_DS/BD rakuten_challengeData/images/images/image_train"

# Load and reduce dataset to 4000 samples
df = pd.read_csv(csv_path).dropna(subset=['designation', 'image_path', 'prdtypecode'])
df = df.sample(n=4000, random_state=42).reset_index(drop=True)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['prdtypecode'])

# Text preprocessing and Word2Vec embedding
sentences = [nlp(text.lower()) for text in df['designation']]
tokenized = [[token.text for token in doc if not token.is_stop and token.is_alpha] for doc in sentences]

w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)

def get_text_embedding(text):
    tokens = [token.text for token in nlp(text.lower()) if not token.is_stop and token.is_alpha]
    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)

# Dataset
class MultimodalDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(image_folder, row['image_path'])
        image = self.transform(Image.open(image_path).convert("RGB"))
        text_emb = get_text_embedding(row['designation'])
        label = row['label']
        return image, torch.tensor(text_emb, dtype=torch.float32), label

# U-Net like model for image embedding
class SimpleUNet(nn.Module):
    def __init__(self, out_features=128):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 32 * 32, out_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

# Prepare dataset
dataset = MultimodalDataset(df)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Extract features for XGBoost
unet = SimpleUNet().to(device)
unet.eval()

image_features = []
text_features = []
labels = []

with torch.no_grad():
    for images, texts, lbls in dataloader:
        images = images.to(device)
        image_emb = unet(images).cpu().numpy()
        image_features.append(image_emb)
        text_features.append(texts.numpy())
        labels.append(lbls.numpy())

X_img = np.vstack(image_features)
X_txt = np.vstack(text_features)
X = np.hstack([X_img, X_txt])
y = np.hstack(labels)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# XGBoost classifier
xgb = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)


# Entraînement XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)

# Précision sur la validation
val_accuracy = xgb.score(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Prédictions sur les données de validation
y_pred = xgb.predict(X_val)

# F1-score pondéré
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1-score pondéré (weighted): {f1:.4f}")
print(classification_report(y_val, y_pred, digits=4))



# SHAP for interpretability, choix sous echantillon 100 elements
explainer = shap.Explainer(xgb)
shap_values = explainer(X_val[:100])
shap.summary_plot(shap_values, X_val[:100], feature_names=[f"img_{i}" for i in range(X_img.shape[1])] + [f"txt_{i}" for i in range(X_txt.shape[1])])
