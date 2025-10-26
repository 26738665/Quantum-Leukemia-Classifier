import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# --- 1. Load the Preprocessed Data ---
DATA_DIR = 'data'
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'leukemia_data_processed.npz')
MODEL_SAVE_PATH = 'classical_leukemia_model.keras'

# Check if the processed data file exists before trying to load it
if not os.path.exists(PROCESSED_DATA_FILE):
    print(f"❌ ERROR: Processed data file not found at '{PROCESSED_DATA_FILE}'")
    print("Please run the 'build_dataset.py' script first to generate the data.")
else:
    print(f"✅ Loading preprocessed data from '{PROCESSED_DATA_FILE}'...")
    with np.load(PROCESSED_DATA_FILE) as data:
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # --- 2. Define the CNN Model Architecture ---
    model = keras.Sequential(
        [
            keras.Input(shape=(128, 128, 3)),

            # Convolutional Block 1: Learns basic features like edges and textures
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Convolutional Block 2: Learns more complex features from the previous layer
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Flattening: Converts the 2D feature maps into a 1D vector
            layers.Flatten(),

            # A dense layer for higher-level reasoning
            layers.Dense(100, activation="relu"),

            # Dropout: A technique to prevent the model from just memorizing the training data
            layers.Dropout(0.5),

            # Output Layer: A single neuron that makes the final prediction (0 or 1)
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Print a summary of the model's structure
    model.summary()

    # --- 3. Compile the Model ---
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # --- 4. Train the Model ---
    print("\n⏳ Starting model training (this will take several minutes)...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=64,  # Increased batch size for more stable training on a larger dataset
        epochs=15,  # Train for 15 full cycles
        validation_data=(X_val, y_val)
    )
    print("\n✅ Model training complete.")

    # --- 5. Evaluate the Model ---
    print("\n🧪 Evaluating model on the unseen test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\n==============================================")
    print(f"  Final Test Accuracy: {test_accuracy:.4f}")
    print(f"  Final Test Loss:     {test_loss:.4f}")
    print(f"==============================================")

    # --- 6. Save the Trained Model ---
    print(f"\n💾 Saving trained model to '{MODEL_SAVE_PATH}'...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully.")

    # --- 7. Plot Training History (Optional) ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("✅ Training history plot saved as 'training_history.png'")
    plt.show()