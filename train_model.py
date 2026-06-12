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
DATA_POINTS_PER_FRAME = 126  # UPGRADED: 2 hands * 21 landmarks * 3 coordinates (x, y, z)

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

# === UPGRADED SPATIAL NORMALIZATION ENGINE ===
def normalize_frame(frame_coordinates):
    """
    Independently normalizes each hand's coordinate matrix block relative 
    to its own respective wrist position to preserve geometric shapes.
    """
    normalized = []
    
    # Hand 1 Normalization (First 63 values)
    wrist1_x = frame_coordinates[0]
    wrist1_y = frame_coordinates[1]
    wrist1_z = frame_coordinates[2]
    
    for i in range(0, 63, 3):
        normalized.append(frame_coordinates[i] - wrist1_x)
        normalized.append(frame_coordinates[i+1] - wrist1_y)
        normalized.append(frame_coordinates[i+2] - wrist1_z)
        
    # Hand 2 Normalization (Next 63 values)
    wrist2_x = frame_coordinates[63]
    wrist2_y = frame_coordinates[64]
    wrist2_z = frame_coordinates[65]
    
    # Safety Check: If hand 2 is zero-padded (invisible), leave the zeros as they are
    if wrist2_x == 0.0 and wrist2_y == 0.0 and wrist2_z == 0.0:
        normalized.extend([0.0] * 63)
    else:
        for i in range(63, 126, 3):
            normalized.append(frame_coordinates[i] - wrist2_x)
            normalized.append(frame_coordinates[i+1] - wrist2_y)
            normalized.append(frame_coordinates[i+2] - wrist2_z)
            
    return normalized

# === DYNAMIC SEQUENCE INTERPOLATION ENGINE ===
def interpolate_sequence(sequence, target_frames=30):
    """
    Dynamically resamples an uneven sequence of coordinate frames
    to a fixed frame count using 1D linear interpolation curves.
    """
    current_frames = len(sequence)
    if current_frames == target_frames:
        return np.array(sequence)
        
    sequence = np.array(sequence)
    original_timesteps = np.linspace(0, 1, current_frames)
    target_timesteps = np.linspace(0, 1, target_frames)
    
    interpolated_matrix = np.zeros((target_frames, sequence.shape[1]))
    
    # Interpolate each of the 126 landmark features independently across time
    for feature_idx in range(sequence.shape[1]):
        interpolated_matrix[:, feature_idx] = np.interp(
            target_timesteps, 
            original_timesteps, 
            sequence[:, feature_idx]
        )
    return interpolated_matrix

# === DATA UNWRAPPING ENGINE WITH BACKWARD COMPATIBILITY ===
def load_real_dataset():
    sequences, labels = [], []
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] '{DATA_PATH}' directory not found!")
        return None, None

    json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
    print(f"Found {len(json_files)} total files in dataset directory.")

    skipped_files_count = 0
    legacy_converted_count = 0

    for file_name in json_files:
        prefix = file_name.split('_')[0]
        if prefix not in label_map:
            continue
            
        file_path = os.path.join(DATA_PATH, file_name)
        
        try:
            with open(file_path, 'r') as f:
                raw_frames = json.load(f)
                
            window_sequences = []
            is_legacy_file = False
            
            # Loop through each frame in the raw file
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
                    
                    # If it's a single nested array containing object splits [{x,y,z}, ...]
                    elif isinstance(target_element, dict) and 'x' in target_element:
                        for landmark in frame:
                            if isinstance(landmark, dict) and 'x' in landmark:
                                frame_coordinates.extend([landmark['x'], landmark['y'], landmark['z']])
                
                # Context Layer 2: Direct fallback if data is structured flat without list wrappers
                elif isinstance(frame, dict) and 'x' in frame:
                    frame_coordinates.extend([frame['x'], frame['y'], frame['z']])

                # BACKWARD COMPATIBILITY ADAPTER INTERCEPT:
                # If this frame unboxed only 63 coordinates, append 63 static zeros to upscale it to 126
                if len(frame_coordinates) == 63:
                    frame_coordinates.extend([0.0] * 63)
                    is_legacy_file = True

                # Check if this single frame successfully matches the active 126 feature boundary line
                if len(frame_coordinates) == DATA_POINTS_PER_FRAME:
                    # Apply dual spatial normalization to lock down positional variance
                    normalized_coords = normalize_frame(frame_coordinates)
                    window_sequences.append(normalized_coords)
            
            # Accept relaxed frame bounds (10 to 60) and interpolate them to exactly 30 frames
            if 10 <= len(window_sequences) <= 60:
                final_sequence = interpolate_sequence(window_sequences, target_frames=SEQUENCE_LENGTH)
                sequences.append(final_sequence)
                labels.append(label_map[prefix])
                if is_legacy_file:
                    legacy_converted_count += 1
            else:
                print(f"[DATA WARNING] File {file_name} has out-of-bounds frame length ({len(window_sequences)}). Skipping.")
                skipped_files_count += 1
                
        except Exception as e:
            print(f"[WARNING] Skipping file {file_name} due to unexpected error: {e}")

    print(f"\n[PARSING SUMMARY] Successfully loaded {len(sequences)} sequences.")
    print(f"-> Legacy 1-Hand Files Adapted via Zero-Padding: {legacy_converted_count}")
    print(f"-> Skipped unparseable/out-of-bounds entries: {skipped_files_count}")
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
# Updated input_shape layer to dynamically accept the 126 dimension feature layout
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, DATA_POINTS_PER_FRAME)),
    Dropout(0.2),
    max_num_hands_lstm := LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print(f"\nBeginning real training epochs on your custom dual-hand landmarks for {len(ACTIONS)} words...")

model.fit(X_train, y_train, epochs=80, batch_size=8, validation_data=(X_val, y_val))

# Export the upgraded intelligent engine
MODEL_NAME = 'mudra_lstm_model.h5'
model.save(MODEL_NAME)
print(f"\n[SUCCESS] Model successfully trained on multi-hand array bounds and exported as '{MODEL_NAME}'!")