# CNN Model for tf_flowers Dataset with Evaluation and Learning Curves

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset (70% train, 30% test)
(train_images, train_labels), (test_images, test_labels) = tfds.load(
    "tf_flowers",
    split=["train[:70%]", "train[:30%]"],
    batch_size=-1,
    as_supervised=True
)

# Resize and normalize
IMG_SIZE = (150, 150)
NUM_CLASSES = 5

train_images = tf.image.resize(train_images, IMG_SIZE) / 255.0
test_images = tf.image.resize(test_images, IMG_SIZE) / 255.0

# One-hot encode labels
y_train = to_categorical(train_labels, num_classes=NUM_CLASSES)
y_test = to_categorical(test_labels, num_classes=NUM_CLASSES)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=10, activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(3),
    tf.keras.layers.Conv2D(32, kernel_size=8, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(32, kernel_size=6, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_images, y_train,
    validation_data=(test_images, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

# Predict
y_pred_probs = model.predict(test_images)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Learning curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# AUC score (macro)
auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr")
print(f"Macro AUC Score: {auc:.4f}")
