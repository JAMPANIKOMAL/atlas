# =============================================================================
# ATLAS - COI PIPELINE - SCRIPT 2: TRAIN MODEL (FINAL SOLUTION)
# =============================================================================
#
# FINAL VERSION - SCALABILITY FIX:
#   -   Implements a custom Keras Data Generator (`COISparseDataGenerator`).
#       This is the industry-standard solution for training on datasets that
#       are too large to fit into GPU VRAM. It works by streaming data from
#       the disk in small batches, ensuring memory usage remains low and
#       stable regardless of the total dataset size. This definitively
#       solves the GPU's "ResourceExhaustedError".
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
# --- FINAL FIX: Import the Sequence class for our data generator ---
from tensorflow.keras.utils import Sequence
import math

# --- Configuration ---
project_root = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"
EPOCHS = 50
BATCH_SIZE = 16
RANDOM_STATE = 42

# --- FINAL FIX: Custom Data Generator Class ---
class COISparseDataGenerator(Sequence):
    """
    A memory-efficient data generator that loads sparse data from disk
    in batches to feed into a Keras model.
    """
    def __init__(self, x_path, y_path, batch_size, num_features, num_classes):
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Load the entire sparse matrix and labels into RAM (not GPU VRAM)
        # This is feasible as system RAM is much larger than VRAM.
        self.x = load_npz(self.x_path)
        self.y = np.load(self.y_path)
        
    def __len__(self):
        """Returns the number of batches per epoch."""
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        """Generates one batch of data."""
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        
        # Get the sparse slice for the batch
        batch_x_sparse = self.x[start_idx:end_idx]
        
        # Convert only this small sparse batch to a dense array
        batch_x_dense = batch_x_sparse.toarray()
        
        batch_y = self.y[start_idx:end_idx]
        
        return batch_x_dense, batch_y

# --- Custom Callback for Clean Output (remains the same) ---
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
    # --- Step 1: Load Metadata (not the full data) ---
    print("--- Step 1: Loading Pre-processed COI Data ---")
    X_train_path = PROCESSED_DATA_DIR / "X_train_coi.npz"
    y_train_path = PROCESSED_DATA_DIR / "y_train_coi.npy"
    X_test_path = PROCESSED_DATA_DIR / "X_test_coi.npz"
    y_test_path = PROCESSED_DATA_DIR / "y_test_coi.npy"

    with open(MODELS_DIR / "coi_genus_label_encoder.pkl", 'rb') as f:
        label_encoder = pickle.load(f)

    # We need the shapes for the generator and model, which we can get from a dummy load
    X_train_shape_ref = load_npz(X_train_path)
    num_training_samples = X_train_shape_ref.shape[0]
    num_features = X_train_shape_ref.shape[1]
    del X_train_shape_ref # Free up memory
    
    print(f"Found {num_training_samples} training samples.")

    # --- Step 2: Define and Compile Model ---
    print("\n--- Step 2: Defining COI Model Architecture ---")
    num_classes = len(label_encoder.classes_)
    input_shape = (num_features,) # Note: input_shape is a tuple
    model = Sequential([
        Dense(2048, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- Step 3: Create Data Generators and Train Model ---
    print("\n--- Step 3: Preparing Data Generators and Starting Training ---")
    
    # Split the training data indices to create validation set
    train_indices, val_indices = train_test_split(np.arange(num_training_samples), test_size=0.1, random_state=RANDOM_STATE)
    
    # We need to load the full data once to split it
    X_train_full = load_npz(X_train_path)
    y_train_full = np.load(y_train_path)

    X_train_final = X_train_full[train_indices]
    y_train_final = y_train_full[train_indices]
    X_val_final = X_train_full[val_indices]
    y_val_final = y_train_full[val_indices]

    # Save these smaller splits back to disk for the generator
    save_npz(PROCESSED_DATA_DIR / "X_train_final_coi.npz", X_train_final)
    np.save(PROCESSED_DATA_DIR / "y_train_final_coi.npy", y_train_final)
    save_npz(PROCESSED_DATA_DIR / "X_val_coi.npz", X_val_final)
    np.save(PROCESSED_DATA_DIR / "y_val_coi.npy", y_val_final)
    
    del X_train_full, y_train_full # Free up memory

    # Instantiate the generators
    training_generator = COISparseDataGenerator(PROCESSED_DATA_DIR / "X_train_final_coi.npz", PROCESSED_DATA_DIR / "y_train_final_coi.npy", BATCH_SIZE, num_features, num_classes)
    validation_generator = COISparseDataGenerator(PROCESSED_DATA_DIR / "X_val_coi.npz", PROCESSED_DATA_DIR / "y_val_coi.npy", BATCH_SIZE, num_features, num_classes)
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    
    # --- FINAL FIX: Train the model using the generator ---
    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, TrainingProgressCallback()],
        workers=4 # Use multiple CPU cores to load data
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
    
    # Create a generator for the test set for evaluation
    test_generator = COISparseDataGenerator(X_test_path, y_test_path, BATCH_SIZE, num_features, num_classes)
    print("  - Evaluating model on the unseen test set...")
    loss, accuracy = loaded_model.evaluate(test_generator, verbose=1)
    
    print("\n" + "-"*42)
    print("--- Final COI Model Evaluation ---")
    print(f"  - Test Set Loss:     {loss:.4f}")
    print(f"  - Test Set Accuracy: {accuracy:.2%}")
    print("-"*42)
    print("\n" + "="*50)
    print("         COI PIPELINE SCRIPT COMPLETE")
    print("="*50)

