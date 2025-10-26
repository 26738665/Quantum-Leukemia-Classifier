import os

os.environ['AUTORAY_BACKEND'] = 'numpy'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pennylane as qml
import matplotlib.pyplot as plt
import time

# --- 1. Setup and Configuration ---
DATA_DIR = 'data'
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'leukemia_data_processed.npz')
CLASSICAL_MODEL_PATH = 'classical_leukemia_model.keras'
HYBRID_MODEL_SAVE_PATH = 'hybrid_quantum_leukemia_model_v2.keras'

# === UPGRADE 1: MORE QUANTUM POWER ===
N_QUBITS = 4
N_LAYERS = 6  # Increased from 3 to 6
# ====================================

dev = qml.device("default.qubit", wires=N_QUBITS)

# --- 2. Load the Preprocessed Data ---
print("✅ Loading all preprocessed data...")
with np.load(PROCESSED_DATA_FILE) as data:
    # === UPGRADE 2: MORE TRAINING DATA ===
    X_train, y_train = data['X_train'][:1000], data['y_train'][:1000]  # Increased from 200 to 1000
    X_val, y_val = data['X_val'][:200], data['y_val'][:200]  # Increased from 50 to 200
    # =======================================
    X_test, y_test = data['X_test'], data['y_test']

print(f"Using {len(X_train)} training samples and {len(X_val)} validation samples.")
print(f"Using {len(X_test)} unseen test samples for final evaluation.")


# --- 3. Define the Quantum Circuit ---
@qml.qnode(dev, interface='tf')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


# --- 4. Build the Hybrid Model ---
print("\nBuilding the Hybrid Quantum-Classical Model v2...")


def create_hybrid_model():
    from tensorflow.keras import layers

    classical_model = keras.models.load_model(CLASSICAL_MODEL_PATH)
    feature_extractor = keras.Model(
        inputs=classical_model.inputs,
        outputs=classical_model.get_layer('dense').output,
        name="feature_extractor"
    )
    feature_extractor.trainable = False

    quantum_layer = qml.qnn.KerasLayer(
        quantum_circuit,
        weight_shapes={"weights": (N_LAYERS, N_QUBITS)},
        output_dim=1,
        name="quantum_layer"
    )

    hybrid_model = keras.Sequential([
        keras.Input(shape=(128, 128, 3)),
        feature_extractor,
        layers.Dense(N_QUBITS, activation="relu", name="adapter_layer"),
        quantum_layer,
        layers.Activation("sigmoid", name="sigmoid_output")
    ])

    # === UPGRADE 3: SMARTER TRAINING ===
    optimizer = keras.optimizers.Adam(learning_rate=0.005)  # Reduced from 0.01 to 0.005
    # ===================================

    hybrid_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return hybrid_model


hybrid_model = create_hybrid_model()
hybrid_model.summary()

# --- 5. Train the Hybrid Model ---
print("\n⏳ Starting hybrid model training (this will be much slower than before)...")
start_time = time.time()  # Let's time how long it takes

history = hybrid_model.fit(
    X_train, y_train, batch_size=8, epochs=10, validation_data=(X_val, y_val))

end_time = time.time()
print(f"\n✅ Hybrid model training complete. Total time: {end_time - start_time:.2f} seconds.")

# --- 6. Evaluate the Model on the Unseen Test Data ---
print("\n🧪 Evaluating hybrid model v2 on the unseen test set...")
test_loss, test_accuracy = hybrid_model.evaluate(X_test, y_test, batch_size=8)

print(f"\n==============================================")
print(f"  Final Hybrid v2 Test Accuracy: {test_accuracy:.4f}")
print(f"==============================================")

# --- 7. Save the Trained Hybrid Model ---
print(f"\n💾 Saving trained hybrid model to '{HYBRID_MODEL_SAVE_PATH}'...")
hybrid_model.save(HYBRID_MODEL_SAVE_PATH)
print("✅ Hybrid model v2 saved successfully.")

# --- 8. Plot Training History ---
# (Plotting code is unchanged)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Hybrid Model v2 - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Hybrid Model v2 - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('hybrid_training_history_v2.png')
print("✅ Training history plot saved as 'hybrid_training_history_v2.png'")
plt.show()