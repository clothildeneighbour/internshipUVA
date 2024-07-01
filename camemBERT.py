#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



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
    def __init__(self, num_labels, epochs=5, lr=1e-5, batch_size=8, weights=None):
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

    def fit(self, X_train, y_train, X_val, y_val):
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

            val_loss = self.evaluate(loader_val)
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

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs.logits, inputs['labels'])
                total_loss += loss.item()
        return total_loss / len(loader)

    def predict(self, X_test):
        self.model.eval()
        inputs = self.tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting', leave=False):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
        self.preds = np.array(all_preds)
        return(self.preds)
    
    def predict_proba(self, X_test): #Méthode si on veut des probas au lieu de classification directe
        self.model.eval()
        inputs = self.tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting', leave=False):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
        self.probs = np.array(all_probs)
        self.preds_proba = np.argmax(self.probs, axis=1)



    def score(self, y_test):
        accuracy = accuracy_score(y_test, self.preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, self.preds, average='weighted')
        return accuracy, precision, recall, f1

    def confusion_matrix(self, y_test):
        cm = confusion_matrix(y_test, self.preds)
        return cm

    def plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()