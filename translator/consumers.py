import os
import json
import numpy as np
import tensorflow as tf
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from django.utils import timezone
from .utils import get_dynamic_actions

# Resolve absolute paths safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'mudra_lstm_model.h5')

# Core Global Memory States
model = None
ACTIONS = []
last_loaded_timestamp = 0  # Tracks the modification time of the file in memory

# ========================================================
# UPGRADED GLOBAL PARAMETERS
# ========================================================
DATA_POINTS_PER_FRAME = 126  # Expected dimension size for simultaneous 2-hand processing

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

def hot_load_model_if_updated():
    """
    Checks the hard drive to see if a background training script 
    has updated the model file. If so, it hot-swaps it live in RAM.
    """
    global model, ACTIONS, last_loaded_timestamp
    
    if not os.path.exists(MODEL_PATH):
        # Even if the model isn't built yet, ensure actions are dynamically populated
        ACTIONS = get_dynamic_actions()
        return

    try:
        # Check the exact last-modified timestamp of the file on disk
        current_disk_timestamp = os.path.getmtime(MODEL_PATH)
        
        # If the file on disk is newer than our loaded memory state, hot-swap it!
        if current_disk_timestamp > last_loaded_timestamp:
            print("\n[MLOPS HOT-SWAP] New weights detected on disk! Reloading layers...")
            
            # 1. Re-compile the neural network architecture from disk safely
            model = tf.keras.models.load_model(MODEL_PATH)
            
            # 2. Rescan the dataset folder to pull the expanded word dictionary labels
            ACTIONS = get_dynamic_actions()
            
            # 3. Synchronize timestamps so we don't reload again until the next train completes
            last_loaded_timestamp = current_disk_timestamp
            
            print(f"[ENGINE LEXICON] Live Interpreter updated with categories: {ACTIONS}\n")
    except Exception as e:
        print(f"[MLOPS HOT-SWAP CRASH] Error loading updated model snapshot: {e}")


class TranslationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        
        # Global boot call moved safely inside the active socket connection thread
        await sync_to_async(hot_load_model_if_updated)()
        
        from .models import TranslationSession
        
        # Look for an existing session from the last 5 minutes that didn't close cleanly
        existing_session = await sync_to_async(
            lambda: TranslationSession.objects.filter(end_time__isnull=True).order_by('-start_time').first()
        )()
        
        if existing_session:
            self.session = existing_session
            print(f"[DB SESSION] Reusing existing open session context row.")
        else:
            self.session = await sync_to_async(TranslationSession.objects.create)()
            print(f"[DB SESSION] Initializing a brand-new translation session.")
            
        self.last_logged_word = None
        
        # Calculate dynamic display index accurately (Option B)
        active_session_index = await sync_to_async(TranslationSession.objects.count)()
        
        await self.send(text_data=json.dumps({
            'session_id': active_session_index if active_session_index > 0 else 1
        }))

    async def disconnect(self, close_code):
        # Leave session open so navigating back and forth preserves history log stacks
        pass

    async def receive(self, text_data):
        global model, ACTIONS
        from .models import TranslationSession, TranslationLog
        
        try:
            data = json.loads(text_data)
            command = data.get('command')
            
            # Explicit termination from UI dashboard buttons
            if command == "end_session":
                if hasattr(self, 'session') and self.session:
                    self.session.end_time = timezone.now()
                    await sync_to_async(self.session.save)()
                    print(f"[DB SESSION] Explicitly closed session via dashboard command.")
                return

            if command == "start_new_session":
                self.session = await sync_to_async(TranslationSession.objects.create)()
                self.last_logged_word = None
                active_session_index = await sync_to_async(TranslationSession.objects.count)()
                await self.send(text_data=json.dumps({'session_id': active_session_index if active_session_index > 0 else 1}))
                return

            # Frame Processing Engine Block
            sequence = data.get('coordinates', [])
            
            # Validation Pass: Ensure incoming multi-hand streams have been processed completely
            if len(sequence) != 30:
                return

            # Periodically check disk timestamp safely mid-stream
            await sync_to_async(hot_load_model_if_updated)()

            # If model isn't built yet, show active lexicon targets as a fallback warning status
            if model is None:
                await self.send(text_data=json.dumps({
                    'prediction': f"Awaiting Data Studio (Target: {ACTIONS})", 
                    'confidence': 0.0
                }))
                return

            # Normalize the live incoming sequence frame-by-frame before hitting the model
            normalized_sequence = []
            for frame in sequence:
                if len(frame) == DATA_POINTS_PER_FRAME:
                    normalized_sequence.append(normalize_frame(frame))
                else:
                    return

            # Feed the upgraded 126-feature spatial matrix directly into hot-swapped LSTM layers
            input_data = np.expand_dims(normalized_sequence, axis=0)
            prediction_scores = model.predict(input_data, verbose=0)[0]

            # ========================================================
            # PURE DYNAMIC LEXICON MASKING ENGINE
            # ========================================================
            # Peek at the very last frame in your 30-frame sequence
            last_frame = normalized_sequence[-1]
            h2_wrist_x, h2_wrist_y, h2_wrist_z = last_frame[63], last_frame[64], last_frame[65]
            
            # Check if second hand is present or zero-padded
            is_two_handed_present = not (h2_wrist_x == 0.0 and h2_wrist_y == 0.0 and h2_wrist_z == 0.0)

            for idx, action_label in enumerate(ACTIONS):
                is_two_handed_label = action_label.startswith('2h_')
                
                if is_two_handed_present:
                    # If BOTH hands are visible, suppress standard single-hand predictions
                    if not is_two_handed_label:
                        prediction_scores[idx] = 0.0
                else:
                    # If ONLY ONE hand is visible, suppress '2h_' prefixed predictions
                    if is_two_handed_label:
                        prediction_scores[idx] = 0.0

            # Re-evaluate the best match after applying structural masking filters
            best_match_idx = np.argmax(prediction_scores)
            confidence = float(prediction_scores[best_match_idx])
            # ========================================================

            # === TEMPORARY DEBUG LOGS ===
            print(f"[ROUTING ENGINE] Two Hands Present: {is_two_handed_present} | Masked Best Match: {best_match_idx}")
            print(f"[DEBUG LOG] Confidence: {confidence:.2f} | Last Logged Word State: {self.last_logged_word}")

            # Logging Engine Rules verification (Only proceed if prediction survived masking gates)
            if confidence >= 0.65 and prediction_scores[best_match_idx] > 0.0 and best_match_idx < len(ACTIONS):
                raw_word = ACTIONS[best_match_idx]
                
                # Strip out the '2h_' routing prefix dynamically for clean logging and display outputs
                resolved_display_word = raw_word.replace('2h_', '')
                print(f"[DEBUG LOG] Resolved Word: '{resolved_display_word}'")
                
                if resolved_display_word != self.last_logged_word and resolved_display_word != 'awaiting_data':
                    print(f"[DB WRITE] Saving '{resolved_display_word}' to Database row...")
                    await sync_to_async(TranslationLog.objects.create)(
                        session=self.session,
                        predicted_word=resolved_display_word,
                        confidence_score=confidence
                    )
                    self.last_logged_word = resolved_display_word
                
                result_word = resolved_display_word
            else:
                # Custom graceful fallback messaging to keep the user informed
                result_word = "Awaiting Two-Hand Dataset..." if is_two_handed_present else "Analyzing..."

            await self.send(text_data=json.dumps({
                'prediction': result_word,
                'confidence': confidence
            }))

        except Exception as e:
            print(f"[STREAM ERROR] Error evaluating frame context: {e}")