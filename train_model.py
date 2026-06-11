import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ========================================================
# CONFIGURATION & HYPERPARAMETERS
# ========================================================
DATA_PATH = os.path.join('dataset')
SEQUENCE_LENGTH = 30  # 30 frames per gesture temporal window
DATA_POINTS_PER_FRAME = 63  # 21 landmarks * 3 coordinates (x, y, z)

ACTIONS = np.array(['one', 'two', 'three'])
label_map = {label: num for num, label in enumerate(ACTIONS)}

def load_real_dataset():
    sequences, labels = [], []
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] '{DATA_PATH}' directory not found!")
        return None, None

    json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
    print(f"Found {len(json_files)} total files in dataset directory.")

    for file_name in json_files:
        prefix = file_name.split('_')[0]
        if prefix not in label_map:
            continue
            
        file_path = os.path.join(DATA_PATH, file_name)
        
        try:
            with open(file_path, 'r') as f:
                raw_frames = json.load(f)
                
            window_sequences = []
            
            # Loop directly through the 30 top-level items (the frames)
            for frame in raw_frames:
                frame_coordinates = []
                
                # Unwrap the second level: grab the inner list of 21 landmarks
                if isinstance(frame, list) and len(frame) > 0:
                    landmarks_list = frame[0]
                    
                    # Loop through the 21 coordinate dictionaries
                    if isinstance(landmarks_list, list):
                        for landmark in landmarks_list:
                            if isinstance(landmark, dict) and 'x' in landmark:
                                frame_coordinates.extend([landmark['x'], landmark['y'], landmark['z']])
                
                # Check if this frame successfully gathered all 63 points
                if len(frame_coordinates) == DATA_POINTS_PER_FRAME:
                    window_sequences.append(frame_coordinates)
            
            # Verify the sequence matches our exact 30-frame window requirement
            if len(window_sequences) == SEQUENCE_LENGTH:
                sequences.append(window_sequences)
                labels.append(label_map[prefix])
            else:
                print(f"[NOTE] File {file_name} only parsed {len(window_sequences)}/30 valid frames. Skipping.")
                
        except Exception as e:
            print(f"[WARNING] Skipping file {file_name} due to unexpected error: {e}")

    return np.array(sequences), np.array(labels)

# ========================================================
# DATA PROCESSING RUNTIME
# ========================================================
print("Starting dataset parsing pipeline...")
X, y = load_real_dataset()

if X is None or len(X) == 0:
    print("[CRITICAL] No valid coordinate sequences could be parsed. Check your dataset format!")
    exit()

print(f"\nSuccessfully processed dataset!")
print(f"Features array (X) shape: {X.shape}") # Should be (Num_Files, 30, 63)
print(f"Labels array (y) shape: {y.shape}")     # Should be (Num_Files,)

# One-hot encode the categorical labels
y_categorical = to_categorical(y, num_classes=len(ACTIONS))

# Split into training and testing configurations (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# ========================================================
# NEURAL NETWORK TRAINING Pipeline
# ========================================================
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, DATA_POINTS_PER_FRAME)),
    Dropout(0.2),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("\nBeginning real training epochs on your custom landmarks...")
model.fit(X_train, y_train, epochs=40, batch_size=4, validation_data=(X_val, y_val))

# Export the intelligent engine
MODEL_NAME = 'mudra_lstm_model.h5'
model.save(MODEL_NAME)
print(f"\n[SUCCESS] Model successfully trained on real data and exported as '{MODEL_NAME}'!")