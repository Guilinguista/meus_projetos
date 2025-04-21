# Transfer Learning with Metrics, Callbacks, and Learning Curves

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score

# Load dataset (replace with your custom dataset loading logic if needed)
(train_ds, val_ds), class_names = tf.keras.utils.image_dataset_from_directory(
    "your_dataset_path",
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(160, 160),
    batch_size=32
), None

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Load pre-trained base model (MobileNetV2 as an example)
IMG_SHAPE = (160, 160, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# Add classifier head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    base_model,
    global_average_layer,
    prediction_layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)

# Train feature extraction
initial_epochs = 5
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=initial_epochs,
                    callbacks=[early_stop, checkpoint])

# Unfreeze the base model for fine-tuning
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_ds,
                         validation_data=val_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         callbacks=[early_stop, checkpoint])

# Load best model
model.load_weights("best_model.h5")

# Evaluate predictions
val_images, val_labels = [], []
for x, y in val_ds.unbatch():
    val_images.append(x.numpy())
    val_labels.append(y.numpy())
val_images = np.array(val_images)
val_labels = np.array(val_labels)

val_preds_logits = model.predict(val_images)
val_preds = tf.nn.sigmoid(val_preds_logits).numpy().flatten()
val_binary = (val_preds > 0.5).astype(int)

# Metrics
print("Classification Report:")
print(classification_report(val_labels, val_binary))

print("Confusion Matrix:")
print(confusion_matrix(val_labels, val_binary))

print(f"AUC: {roc_auc_score(val_labels, val_preds):.4f}")
print(f"Precision: {precision_score(val_labels, val_binary):.4f}")
print(f"Recall: {recall_score(val_labels, val_binary):.4f}")

# Plot learning curves
def plot_learning(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.tight_layout()
    plt.show()

plot_learning(history)
plot_learning(history_fine)
