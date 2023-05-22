import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dataset class
class IntentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define the classifier model
class ClassifierModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierModel, self).__init__()
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert_encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Load data
trainfile = datafolder + "train.csv"
testfile = datafolder + "test.csv"
validfile = datafolder + "valid.csv"

traindf = pd.read_csv(trainfile)
validdf = pd.read_csv(validfile)
testdf = pd.read_csv(testfile)

trainfeatures = traindf["text"].values
trainlabels = traindf["intent"].values
binarizer = LabelBinarizer()
trainlabels = binarizer.fit_transform(trainlabels)

testfeatures = testdf["text"].values
testlabels = testdf["intent"].values
validfeatures = validdf["text"].values
validlabels = validdf["intent"].values

# Tokenize input sequences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(trainfeatures.tolist(), truncation=True, padding=True, return_tensors='pt')
train_labels = torch.tensor(trainlabels, dtype=torch.float32)

test_encodings = tokenizer(testfeatures.tolist(), truncation=True, padding=True, return_tensors='pt')
test_labels = torch.tensor(binarizer.transform(testlabels), dtype=torch.float32)

valid_encodings = tokenizer(validfeatures.tolist(), truncation=True, padding=True, return_tensors='pt')
valid_labels = torch.tensor(binarizer.transform(validlabels), dtype=torch.float32)

# Create dataloaders
batch_size = 32

train_dataset = IntentDataset(train_encodings, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = IntentDataset(test_encodings, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

valid_dataset = IntentDataset(valid_encodings, valid_labels)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# Create an instance of the model
model = ClassifierModel(num_classes=7)
model.to(device)

# Define optimizer, loss function, and metrics
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.BCEWithLogitsLoss()
metrics = {'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred.round(), average='micro'),
           'precision': lambda y_true, y_pred: precision_score(y_true, y_pred.round(), average='micro'),
           'recall': lambda y_true, y_pred: recall_score(y_true, y_pred.round(), average='micro')}

# Training loop
epochs = 5
for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)

    # Evaluation on validation set
    model.eval()
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    with torch.no_grad():
        for inputs, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            f1 = metrics['f1_score'](labels, outputs)
            precision = metrics['precision'](labels, outputs)
            recall = metrics['recall'](labels, outputs)

            total_f1 += f1
            total_precision += precision
            total_recall += recall

    avg_f1 = total_f1 / len(valid_dataloader)
    avg_precision = total_precision / len(valid_dataloader)
    avg_recall = total_recall / len(valid_dataloader)

    print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - F1: {avg_f1:.4f} - Precision: {avg_precision:.4f} - Recall: {avg_recall:.4f}')

# Testing
model.eval()
total_f1 = 0
total_precision = 0
total_recall = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        f1 = metrics['f1_score'](labels, outputs)
        precision = metrics['precision'](labels, outputs)
        recall = metrics['recall'](labels, outputs)

        total_f1 += f1
        total_precision += precision
        total_recall += recall

avg_f1 = total_f1 / len(test_dataloader)
avg_precision = total_precision / len(test_dataloader)
avg_recall = total_recall / len(test_dataloader)

print(f'Test Results - F1: {avg_f1:.4f} - Precision: {avg_precision:.4f} - Recall: {avg_recall:.4f}')
