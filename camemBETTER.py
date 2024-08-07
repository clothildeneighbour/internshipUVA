#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:38:20 2024

@author: cocoticota
"""


from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class CustomLoss(torch.nn.Module):
    def __init__(self, weights):
        super(CustomLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, labels):
        loss = F.cross_entropy(outputs, labels, reduction='none')
        for i in range(len(labels)):
            if labels[i] == 0:
                loss[i] *= self.weights[0]
        return loss.mean()

class CamembertModel():
    def __init__(self, num_labels, epochs=5, lr=1e-5, batch_size=8, weights=None, labels_map=None):
        self.num_labels = num_labels
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weights = weights if weights is not None else torch.tensor([1.0] * num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        self.model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=num_labels).to(self.device)
        self.model.classifier.dropout = torch.nn.Dropout(0.3)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_fn = CustomLoss(self.weights)
        self.labels_map = labels_map
        self.thresholds = np.ones(num_labels) * 0.5  # Initialisation des seuils à 0.5

    def fit(self, X_train, y_train, X_val, y_val):
        check_labels(torch.tensor(y_train), self.num_labels)
        check_labels(torch.tensor(y_val), self.num_labels)
        inputs_train = self.tokenizer(X_train, padding=True, truncation=True, return_tensors='pt')
        dataset_train = TensorDataset(inputs_train['input_ids'], inputs_train['attention_mask'], torch.tensor(y_train))
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        inputs_val = self.tokenizer(X_val, padding=True, truncation=True, return_tensors='pt')
        dataset_val = TensorDataset(inputs_val['input_ids'], inputs_val['attention_mask'], torch.tensor(y_val))
        loader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)

        best_loss = float('inf')
        patience, trials = 3, 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(loader_train, desc=f'Epoch {epoch + 1}/{self.epochs}', leave=False)
            for batch in progress_bar:
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs.logits, inputs['labels'])
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(total_loss / len(progress_bar))})

            val_loss = self.rate(loader_val)
            if val_loss < best_loss:
                best_loss = val_loss
                trials = 0
                # Sauvegarder le meilleur modèle
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                trials += 1
                if trials >= patience:
                    print(f"Early stopping on epoch {epoch + 1}")
                    break


    def rate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                check_labels(batch[2], self.num_labels)
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs.logits, inputs['labels'])
                total_loss += loss.item()
        return total_loss / len(loader)


    def predict(self, X_test, adjusted = False, keywords = None, label_index = None):
        self.model.eval()
        inputs = self.tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_probabilities = []
        with torch.no_grad():
            for batch in tqdm(self.loader, desc='Predicting', leave=False):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)          
                if adjusted : 
                    if keywords is  None or label_index is None : 
                        raise ValueError("Keywords and label_index must not be None if 'adjusted' is True.")
                    else :                  
                        probabilities = torch.tensor(adjust_predictions(X_test, keywords, label_index, probabilities))
                preds = self.apply_thresholds(probabilities)
                all_preds.extend(preds.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        self.preds = np.array(all_preds)
        self.probabilities = np.array(all_probabilities)
        
    def get_prediction(self, X_test) : 
        self.predict(X_test)
        return self.preds
    
    def get_probabilities(self, X_test) : 
        self.predict(X_test)
        return self.probabilities 
    

    def apply_thresholds(self, probabilities):
        preds = torch.zeros(probabilities.shape[0], dtype=torch.long).to(self.device)
        for i in range(probabilities.shape[0]):
            for j in range(self.num_labels):
                if probabilities[i, j] > self.thresholds[j]:
                    preds[i] = j
        return preds

    def set_thresholds(self, thresholds):
        assert len(thresholds) == self.num_labels, "Number of thresholds must match number of labels"
        self.thresholds = np.array(thresholds)
        
        
    def score(self, y_test):
        accuracy = accuracy_score(y_test, self.preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, self.preds, average='weighted')
        return accuracy, precision, recall, f1

    def confusion_matrix(self, y_test):
        cm = confusion_matrix(y_test, self.preds)
        return cm

    def plot_confusion_matrix(self, y_test, labels = None):
        
        cm = confusion_matrix(y_test, self.preds)   
        if labels is None : 
            labels = self.labels_map
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        
        
        
def adjust_predictions(X_test, keywords, label_index, probabilities):
    
    probabilities = np.array(probabilities)
    # Ajustement des probabilités basées sur les mots clés
    for i, title in enumerate(X_test):
        if any(keyword in title.lower() for keyword in keywords):
            probabilities[i][label_index] += 0.5
            probabilities[i][label_index] = min(1.0, probabilities[i][label_index])
            # Normalisation pour s'assurer que la somme des probabilités reste 1
            probabilities[i] = probabilities[i] / probabilities[i].sum()
    return probabilities
            
        
def check_labels(labels, num_labels):
    assert torch.all(labels >= 0) and torch.all(labels < num_labels), "Labels out of range"


def create_balanced_sample(df, class_column, n_per_class):
    # Sélectionner n échantillons de chaque classe
    balanced_sample = df.groupby(class_column).apply(lambda x: x.sample(n=min(n_per_class, len(x)))).reset_index(drop=True)
    return balanced_sample


def labels_to_int(df, label_column) :
    # Convertir les étiquettes en entiers
    label_map = {label: idx for idx, label in enumerate(df[label_column].unique())}
    df['label_id'] = df[label_column].map(label_map)
    return df, label_map


def split_from_df(df, text) :
    ## Diviser les données en ensembles d'entraînement et de test
    X = df[text].tolist()
    y = df['label_id'].tolist()
    # Diviser les données en ensembles d'entraînement et de test/validation
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # Diviser l'ensemble temporaire en ensembles de validation et de test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return(X_train, y_train, X_val, y_val, X_test, y_test)



def preprocess_data(df, class_column, text_column, n = None ,
                    balanced = False, sample = False ) :
    """Fonction qui prend en argument un DataFrame avec une colonne text sur laquelle utiliser
       CamemBERT, le nombre de classes, la colonne avec les classes. Par défaut la
       fonction tire un n échantillon aléatoire de df qui préserve la structure originelle
       des labels. En utilisant balanced = True, l'échantillon comportera n exemples de chaque
       label."""
    if balanced == True :
        df = create_balanced_sample(df, class_column, n)
    elif sample == True :
        _, df = train_test_split(
        df,
        train_size=n,
        stratify=df[class_column],
        random_state=42  # Pour la reproductibilité
    )
    df, label_map = labels_to_int(df, class_column)
    return label_map, split_from_df(df, text_column)





def main(df, class_column, text_column, num_labels, n = None ,
         balanced = False, sample = False,  weights = None ):

    labels_map, train_test_val = preprocess_data(df, class_column, text_column, n,
        balanced, sample)
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_val
    model = CamembertModel(num_labels = num_labels, weights = weights, labels_map = labels_map)
    model.fit(X_train, y_train, X_val, y_val)
      # Définir les seuils pour chaque classe
    return model, y_test, X_test        
        
        
        