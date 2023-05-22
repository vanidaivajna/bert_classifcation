# Predict intents for the preprocessed text data
predictions = model.predict(new_data_preprocessed)
predicted_labels_encoded = np.argmax(predictions, axis=1)

# Convert the predicted encoded labels back to their original labels
predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
from sklearn.metrics import classification_report

# Convert the true labels and predicted labels to their original form
true_labels = label_encoder.inverse_transform(new_data_labels_encoded)
predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)

# Generate the classification report
report = classification_report(true_labels, predicted_labels)
target_names = label_encoder.classes_
report = classification_report(true_labels, predicted_labels, target_names=target_names)

# Print the classification report
print(report)
target_names = label_encoder.classes_
report = classification_report(true_labels, predicted_labels, target_names=target_names)
