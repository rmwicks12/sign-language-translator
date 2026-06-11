import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_PATH = os.path.join('dataset')
SEQUENCE_LENGTH = 30  # 30 frames per gesture temporal window
DATA_POINTS_PER_FRAME = 63  # 21 landmarks * 3 coordinates (x, y, z)

# Define Mudrā's vocabulary classes
ACTIONS = np.array(['hello', 'thank_you', 'yes', 'no'])
label_map = {label: num for num, label in enumerate(ACTIONS)}

# ==========================================
# DATA LOADING OR MOCK GENERATION
# ==========================================
def load_dataset():
    sequences, labels = [], []
    real_data_found = False

    # Check if dataset directory exists and has subfolders
    if os.path.exists(DATA_PATH):
        for action in ACTIONS:
            action_path = os.path.join(DATA_PATH, action)
            if os.path.exists(action_path):
                json_files = [f for f in os.listdir(action_path) if f.endswith('.json')]
                if json_files:
                    real_data_found = True
                    print(f"Loading {len(json_files)} real samples for action: '{action}'...")
                    for file_name in json_files:
                        with open(os.path.join(action_path, file_name), 'read') as f:
                            data = json.load(f)
                            # Expecting data structure to be a list of 30 frames, each containing 63 flattened points
                            sequences.append(data)
                            labels.append(label_map[action])

    # Fallback to generating mock data if no real JSONs exist
    if not real_data_found:
        print("\n[!] No real JSON data found. Generating synthetic dataset to test pipeline...")
        SAMPLES_PER_CLASS = 40  # Generate 40 fake recordings per word
        
        for action in ACTIONS:
            for _ in range(SAMPLES_PER_CLASS):
                # Generate a random tensor mimicking a smooth hand movement path
                # Base random coordinates + a small directional drift over 30 frames
                base_motion = np.random.rand(SEQUENCE_LENGTH, DATA_POINTS_PER_FRAME)
                drift = np.linspace(0, 0.2, SEQUENCE_LENGTH).reshape(-1, 1)
                mock_sequence = base_motion + drift
                
                sequences.append(mock_sequence.tolist())
                labels.append(label_map[action])
                
    return np.array(sequences), np.array(labels)

# Load up our data
X, y = load_dataset()

# Preprocess features and convert target labels to one-hot encoded vectors
y = to_categorical(y, num_classes=len(ACTIONS)).astype(int)

# Split dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data processing complete.")
print(f"Training features shape: {X_train.shape} (Samples, Frames, Coordinates)")
print(f"Training labels shape: {y_train.shape}\n")

# ==========================================
# DEFINE MUDRA'S LSTM ARCHITECTURE
# ==========================================
model = Sequential([
    # First LSTM layer captures sequential patterns; returns sequences to feed the next layer
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, DATA_POINTS_PER_FRAME)),
    Dropout(0.2), # Mitigates overfitting
    
    # Second LSTM layer consolidates temporal trajectories into a single state vector
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    
    # Dense layers map temporal features to class categories
    Dense(64, activation='relu'),
    Dense(len(ACTIONS), activation='softmax') # Outputs probability distribution across vocabulary
])

# Compile model with cross-entropy loss function for classification
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# ==========================================
# TRAIN AND SAVE THE MODEL
# ==========================================
print("\nStarting model training...")
# Run for a brief 20 epochs on mock data to check compiler health
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Export trained weights so our Django consumer can pull them for real-time inference
MODEL_NAME = 'mudra_lstm_model.h5'
model.save(MODEL_NAME)
print(f"\n[SUCCESS] Model successfully compiled, trained, and exported as '{MODEL_NAME}'!")