#!/usr/bin/env python3
"""
Train a Tomato Disease Classification Model using TensorFlow and Keras.

This script uses transfer learning with the MobileNetV2 architecture to train
a model on a dataset of tomato leaf images organized by disease class.
"""
import tensorflow as tf
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# --- Configuration ---
# Set up basic logging to see the script's progress.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Paths and Parameters ---
# IMPORTANT: Update this path to the location of your dataset folder.
# Based on your prompt, it should be something like "Desktop/newdata/matixa"
DATASET_DIR = Path("matixa") # <-- UPDATE THIS PATH

# Model training parameters
IMG_SIZE = (224, 224)  # Input image size for MobileNetV2
BATCH_SIZE = 32        # Number of images to process in each batch
LEARNING_RATE = 0.001  # Controls how much the model is updated per batch
EPOCHS = 15            # Number of times to loop over the entire dataset
VALIDATION_SPLIT = 0.2 # Use 20% of the data for validation

# --- File Paths for Saving ---
MODEL_SAVE_PATH = "tomato_disease_classifier.keras"
PLOT_SAVE_PATH = "training_history.png"

def build_model(num_classes: int):
    """
    Builds a transfer learning model using MobileNetV2 as the base.

    Args:
        num_classes: The number of disease classes to predict.

    Returns:
        A compiled Keras model ready for training.
    """
    logger.info(f"Building model for {num_classes} classes...")

    # 1. Load the MobileNetV2 model, pre-trained on ImageNet.
    # `include_top=False` removes the final classification layer.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )

    # 2. Freeze the layers of the base model so they are not re-trained.
    base_model.trainable = False
    logger.info("Base model layers have been frozen.")

    # 3. Add our custom classification layers on top of the base model.
    # The GlobalAveragePooling2D layer converts the features to a 1D vector.
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # A Dropout layer helps prevent overfitting by randomly setting a fraction of inputs to 0.
    x = tf.keras.layers.Dropout(0.5)(x)
    # The final Dense layer with 'softmax' activation outputs a probability for each class.
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # 4. Construct the full model.
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # 5. Compile the model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("✅ Model built and compiled successfully.")
    model.summary()
    return model

def plot_history(history):
    """
    Plots the training and validation accuracy and loss.

    Args:
        history: The History object returned by model.fit().
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(PLOT_SAVE_PATH)
    logger.info(f"📈 Training history plot saved to '{PLOT_SAVE_PATH}'")
    plt.show()

def main():
    """Main function to load data, build the model, train, and save it."""
    if not DATASET_DIR.exists():
        logger.error(f"Dataset directory not found: '{DATASET_DIR}'")
        logger.error("Please ensure the path is correct and you have run the data preparation script.")
        return

    # Load the training dataset from the directory structure
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123, # Use a seed for reproducibility
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Load the validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Get the class names from the directory names
    class_names = train_ds.class_names
    num_classes = len(class_names)
    logger.info(f"Found {num_classes} classes: {class_names}")

    if num_classes == 0:
        logger.error("No image classes found. The dataset directory may be empty or misconfigured.")
        return

    # Configure the dataset for performance
    # .cache() keeps images in memory after they're loaded off disk during the first epoch.
    # .prefetch() overlaps data preprocessing and model execution while training.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Normalize the pixel values from [0, 255] to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Build the model
    model = build_model(num_classes=num_classes)

    # Train the model
    logger.info(f"🚀 Starting model training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    logger.info(f"✅ Training complete. Model saved to '{MODEL_SAVE_PATH}'")

    # Plot the training history
    plot_history(history)

if __name__ == "__main__":
    # Check for TensorFlow installation
    try:
        import tensorflow
    except ImportError:
        logger.error("❌ TensorFlow library not found.")
        logger.error("   Please install it by running: pip install tensorflow")
    else:
        main()
