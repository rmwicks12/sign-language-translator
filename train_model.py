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

# === DYNAMIC LEXICON EXTRACTION FIX ===
def get_dynamic_actions():
    """Scans the dataset directory and dynamically extracts all active vocabulary labels."""
    if not os.path.exists(DATA_PATH):
        return ['one', 'two', 'three']
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
    actions = set()
    for file in files:
        if '_' in file:
            actions.add(file.split('_')[0])
    return sorted(list(actions)) if actions else ['one', 'two', 'three']

# Generate vocabulary map on the fly
ACTIONS = np.array(get_dynamic_actions())
label_map = {label: num for num, label in enumerate(ACTIONS)}

print(f"\n[MLOPS TRAINING] Dynamic vocabulary mapped successfully:")
print(f"Index Mapping: {label_map}\n")

# === SPATIAL NORMALIZATION ENGINE ===
def normalize_frame(frame_coordinates):
    """
    Takes a flat list of 63 coordinates (21 landmarks * 3 axes).
    Anchors the wrist (first 3 values: x, y, z) to (0, 0, 0).
    Calculates all other joints relative to the wrist position.
    """
    wrist_x = frame_coordinates[0]
    wrist_y = frame_coordinates[1]
    wrist_z = frame_coordinates[2]
    
    normalized = []
    for i in range(0, len(frame_coordinates), 3):
        normalized.append(frame_coordinates[i] - wrist_x)
        normalized.append(frame_coordinates[i+1] - wrist_y)
        normalized.append(frame_coordinates[i+2] - wrist_z)
        
    return normalized

# === DATA UNWRAPPING ENGINE ===
def load_real_dataset():
    sequences, labels = [], []
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] '{DATA_PATH}' directory not found!")
        return None, None

    json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
    print(f"Found {len(json_files)} total files in dataset directory.")

    skipped_files_count = 0

    for file_name in json_files:
        prefix = file_name.split('_')[0]
        if prefix not in label_map:
            continue
            
        file_path = os.path.join(DATA_PATH, file_name)
        
        try:
            with open(file_path, 'r') as f:
                raw_frames = json.load(f)
                
            window_sequences = []
            
            # Loop through each of the 30 frames
            for frame in raw_frames:
                frame_coordinates = []
                
                # Context Layer 1: Check if the frame is a nested list wrapper
                if isinstance(frame, list) and len(frame) > 0:
                    target_element = frame[0]
                    
                    # If it's your legacy double-nested list format [[{x,y,z}, ...]]
                    if isinstance(target_element, list):
                        for landmark in target_element:
                            if isinstance(landmark, dict) and 'x' in landmark:
                                frame_coordinates.extend([landmark['x'], landmark['y'], landmark['z']])
                    
                    # If it's a single nested array containing your new object splits [{x,y,z}, ...]
                    elif isinstance(target_element, dict) and 'x' in target_element:
                        for landmark in frame:
                            if isinstance(landmark, dict) and 'x' in landmark:
                                frame_coordinates.extend([landmark['x'], landmark['y'], landmark['z']])
                
                # Context Layer 2: Direct fallback if data is structured flat without list wrappers
                elif isinstance(frame, dict) and 'x' in frame:
                    if isinstance(frame, dict) and 'x' in frame:
                        frame_coordinates.extend([frame['x'], frame['y'], frame['z']])

                # Check if this single frame successfully unboxed all 63 coordinates
                if len(frame_coordinates) == DATA_POINTS_PER_FRAME:
                    # Spatial Normalization layer scales coords before saving to window matrix
                    normalized_coords = normalize_frame(frame_coordinates)
                    window_sequences.append(normalized_coords)
            
            # Verify the sequence matches your exact 30-frame window requirement
            if len(window_sequences) == SEQUENCE_LENGTH:
                sequences.append(window_sequences)
                labels.append(label_map[prefix])
            else:
                print(f"[DATA WARNING] File {file_name} only extracted {len(window_sequences)}/30 valid frames. Skipping.")
                skipped_files_count += 1
                
        except Exception as e:
            print(f"[WARNING] Skipping file {file_name} due to unexpected error: {e}")

    print(f"\n[PARSING SUMMARY] Successfully loaded {len(sequences)} sequences. Skipped {skipped_files_count} incompatible entries.")
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
print(f"Features array (X) shape: {X.shape}") 
print(f"Labels array (y) shape: {y.shape}")     

# Dynamic category length scaling
y_categorical = to_categorical(y, num_classes=len(ACTIONS))

# Split into training and testing configurations (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# ========================================================
# NEURAL NETWORK TRAINING PIPELINE
# ========================================================
# Neural network architecture grows dynamically based on the length of ACTIONS
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, DATA_POINTS_PER_FRAME)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print(f"\nBeginning real training epochs on your custom landmarks for {len(ACTIONS)} words...")

model.fit(X_train, y_train, epochs=80, batch_size=8, validation_data=(X_val, y_val))

# Export the intelligent engine
MODEL_NAME = 'mudra_lstm_model.h5'
model.save(MODEL_NAME)
print(f"\n[SUCCESS] Model successfully trained on real data and exported as '{MODEL_NAME}'!")