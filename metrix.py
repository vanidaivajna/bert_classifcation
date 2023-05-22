loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
    # Add more metrics as needed
]

# Compile the model with the loss and metrics
classifier_model.compile(optimizer='adam', loss=loss, metrics=metrics)
