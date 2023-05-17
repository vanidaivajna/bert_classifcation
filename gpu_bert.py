import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report

# Check the available GPU devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPU devices:", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the dataset
data = pd.read_excel("banking77.xlsx")

# Split the dataset into text and label
texts = data["text"].tolist()
labels = data["intent"].tolist()

# Convert the labels to one-hot encoding
labels = pd.get_dummies(labels)

# Convert the texts and labels to numpy arrays
texts = np.array(texts, dtype=object)[:, np.newaxis]
labels = np.array(labels)

# Split the dataset into training, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels)


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Encode the training, validation, and test texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
valid_encodings = tokenizer(valid_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Convert the encodings and labels to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(len(train_labels)).batch(16)

valid_dataset = tf.data.Dataset.from_tensor_slices((
    dict(valid_encodings),
    valid_labels
)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
)).batch(16)


# Load the BERT model with GPU support
strategy = tf.distribute.OneDeviceStrategy("GPU:0")
with strategy.scope():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=22)

    # Fine-tune the model on the training dataset
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(train_dataset, epochs=3, validation_data=valid_dataset)
# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)

# Generate predictions for the test dataset
test_predictions = model.predict(test_dataset)
test_predictions = np.argmax(test_predictions.logits, axis=1)

# Convert the test labels to their corresponding classes
test_labels = np.argmax(test_labels, axis=1)
