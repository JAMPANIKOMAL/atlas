# =============================================================================
# ATLAS - COI PIPELINE - SCRIPT 2: TRAIN MODEL (FINAL SOLUTION)
# =============================================================================
#
# FINAL VERSION - SCALABILITY FIX:
#   -   Reduces the size of the neural network's dense layers to prevent
#       GPU VRAM exhaustion, which was a primary cause of the errors.
#   -   This new architecture is optimized to work with the smaller,
#       curated 1% sample dataset created by the previous script.
#
# =============================================================================

# --- Imports ---
import numpy as np
import tensorflow as tf
from scipy.sparse import load_npz
import pickle
from pathlib import Path
import gc
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
# --- We no longer need a custom data generator, as the dataset is small now ---
# from tensorflow.keras.utils import Sequence
# import math

# --- Configuration ---
project_root = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"
EPOCHS = 50
BATCH_SIZE = 16
RANDOM_STATE = 42

# --- Custom Callback for Clean Output (unchanged) ---
class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy', 0); val_acc = logs.get('val_accuracy', 0)
        loss = logs.get('loss', 0); val_loss = logs.get('val_loss', 0)
        acc_bar = '█' * int(acc * 20) + '·' * (20 - int(acc * 20))
        val_acc_bar = '█' * int(val_acc * 20) + '·' * (20 - int(val_acc * 20))
        print(f"\rEpoch {epoch+1:02d}/{EPOCHS} | Loss: {loss:.4f} | Acc: {acc:.2%} [{acc_bar}] | Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.2%} [{val_acc_bar}]", end='')

# =============================================================================
# --- Main Script Execution ---
# =============================================================================

if __name__ == "__main__":
    # --- Step 1: Load Data ---
    print("--- Step 1: Loading Pre-processed COI Data ---")
    X_train_path = PROCESSED_DATA_DIR / "X_train_coi.npz"
    y_train_path = PROCESSED_DATA_DIR / "y_train_coi.npy"
    X_test_path = PROCESSED_DATA_DIR / "X_test_coi.npz"
    y_test_path = PROCESSED_DATA_DIR / "y_test_coi.npy"

    try:
        X_train = load_npz(X_train_path)
        X_test = load_npz(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
        with open(MODELS_DIR / "coi_genus_label_encoder.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the `01_prepare_data.py` script first.")
        exit()

    print(f"Loaded {X_train.shape[0]} training samples.")

    # --- Step 2: Define and Compile Model ---
    print("\n--- Step 2: Defining COI Model Architecture ---")
    num_classes = len(label_encoder.classes_)
    input_shape = X_train.shape[1]
    
    # --- REDUCED MODEL ARCHITECTURE ---
    # The original model was too large for your GPU. We have reduced the
    # size of the dense layers to make them fit in memory.
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- Step 3: Train Model ---
    print("\n--- Step 3: Preparing Data and Starting Training ---")
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train)
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    
    history = model.fit(
        X_train_final, y_train_final,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[early_stopping, TrainingProgressCallback()]
    )
    print("\n\n--- Training complete. ---")

    # --- Step 4: Save, Reload, and Evaluate ---
    print("\n--- Step 4: Saving, Reloading, and Evaluating Model ---")
    MODEL_PATH = MODELS_DIR / "coi_genus_classifier.keras"
    model.save(MODEL_PATH)
    print(f"  - Model saved to: {MODEL_PATH}")

    tf.keras.backend.clear_session()
    gc.collect()

    loaded_model = load_model(MODEL_PATH)
    print("  - Evaluating model on the unseen test set...")
    # NOTE: The batch_size parameter is crucial for memory management here.
    loss, accuracy = loaded_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
    
    print("\n" + "-"*42)
    print("--- Final COI Model Evaluation ---")
    print(f"  - Test Set Loss:     {loss:.4f}")
    print(f"  - Test Set Accuracy: {accuracy:.2%}")
    print("-"*42)
    print("\n" + "="*50)
    print("         COI PIPELINE SCRIPT COMPLETE")
    print("="*50)
