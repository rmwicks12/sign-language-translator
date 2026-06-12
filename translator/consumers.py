import os
import json
import numpy as np
import tensorflow as tf
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from django.utils import timezone
from .utils import get_dynamic_actions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'mudra_lstm_model.h5')

model = None
ACTIONS = []
last_loaded_timestamp = 0

DATA_POINTS_PER_FRAME = 126 

def normalize_frame(frame_coordinates):
    normalized = []
    
    wrist1_x = frame_coordinates[0]
    wrist1_y = frame_coordinates[1]
    wrist1_z = frame_coordinates[2]
    
    for i in range(0, 63, 3):
        normalized.append(frame_coordinates[i] - wrist1_x)
        normalized.append(frame_coordinates[i+1] - wrist1_y)
        normalized.append(frame_coordinates[i+2] - wrist1_z)
        
    wrist2_x = frame_coordinates[63]
    wrist2_y = frame_coordinates[64]
    wrist2_z = frame_coordinates[65]
    
    if wrist2_x == 0.0 and wrist2_y == 0.0 and wrist2_z == 0.0:
        normalized.extend([0.0] * 63)
    else:
        for i in range(63, 126, 3):
            normalized.append(frame_coordinates[i] - wrist2_x)
            normalized.append(frame_coordinates[i+1] - wrist2_y)
            normalized.append(frame_coordinates[i+2] - wrist2_z)
            
    return normalized

def hot_load_model_if_updated():
    global model, ACTIONS, last_loaded_timestamp
    
    if not os.path.exists(MODEL_PATH):
        ACTIONS = get_dynamic_actions()
        return

    try:
        current_disk_timestamp = os.path.getmtime(MODEL_PATH)
        if current_disk_timestamp > last_loaded_timestamp:
            print("\n[MLOPS HOT-SWAP] New weights detected on disk! Reloading layers...")
            model = tf.keras.models.load_model(MODEL_PATH)
            ACTIONS = get_dynamic_actions()
            last_loaded_timestamp = current_disk_timestamp
            print(f"[ENGINE LEXICON] Live Interpreter updated with categories: {ACTIONS}\n")
    except Exception as e:
        print(f"[MLOPS HOT-SWAP CRASH] Error loading updated model snapshot: {e}")

class TranslationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await sync_to_async(hot_load_model_if_updated)()
        
        from .models import TranslationSession
        existing_session = await sync_to_async(
            lambda: TranslationSession.objects.filter(end_time__isnull=True).order_by('-start_time').first()
        )()
        
        if existing_session:
            self.session = existing_session
        else:
            self.session = await sync_to_async(TranslationSession.objects.create)()
            
        self.last_logged_word = None
        active_session_index = await sync_to_async(TranslationSession.objects.count)()
        await self.send(text_data=json.dumps({'session_id': active_session_index if active_session_index > 0 else 1}))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        global model, ACTIONS
        from .models import TranslationSession, TranslationLog
        
        try:
            data = json.loads(text_data)
            command = data.get('command')
            
            if command == "end_session":
                if hasattr(self, 'session') and self.session:
                    self.session.end_time = timezone.now()
                    await sync_to_async(self.session.save)()
                return

            if command == "start_new_session":
                self.session = await sync_to_async(TranslationSession.objects.create)()
                self.last_logged_word = None
                active_session_index = await sync_to_async(TranslationSession.objects.count)()
                await self.send(text_data=json.dumps({'session_id': active_session_index if active_session_index > 0 else 1}))
                return

            sequence = data.get('coordinates', [])
            if len(sequence) != 30:
                return

            await sync_to_async(hot_load_model_if_updated)()

            if model is None:
                await self.send(text_data=json.dumps({'prediction': f"Awaiting Data Studio", 'confidence': 0.0}))
                return

            normalized_sequence = []
            for frame in sequence:
                if len(frame) == DATA_POINTS_PER_FRAME:
                    normalized_sequence.append(normalize_frame(frame))
                else:
                    return

            input_data = np.expand_dims(normalized_sequence, axis=0)
            prediction_scores = model.predict(input_data, verbose=0)[0]

            last_frame = normalized_sequence[-1]
            h2_wrist_x, h2_wrist_y, h2_wrist_z = last_frame[63], last_frame[64], last_frame[65]
            is_two_handed_present = not (h2_wrist_x == 0.0 and h2_wrist_y == 0.0 and h2_wrist_z == 0.0)

            for idx, action_label in enumerate(ACTIONS):
                is_two_handed_label = action_label.startswith('2h_')
                if is_two_handed_present:
                    if not is_two_handed_label:
                        prediction_scores[idx] = 0.0
                else:
                    if is_two_handed_label:
                        prediction_scores[idx] = 0.0

            best_match_idx = np.argmax(prediction_scores)
            confidence = float(prediction_scores[best_match_idx])

            if confidence >= 0.65 and prediction_scores[best_match_idx] > 0.0 and best_match_idx < len(ACTIONS):
                raw_word = ACTIONS[best_match_idx]
                resolved_display_word = raw_word.replace('2h_', '')
                
                # === OPTION 1: THE CONTINUOUS RECOGNITION FILTER ===
                if resolved_display_word.lower() in ['neutral', 'transition']:
                    # Reset the lock state. Do NOT save to database.
                    self.last_logged_word = None
                    result_word = "..." # Visually indicate a resting transition
                else:
                    if resolved_display_word != self.last_logged_word and resolved_display_word != 'awaiting_data':
                        await sync_to_async(TranslationLog.objects.create)(
                            session=self.session,
                            predicted_word=resolved_display_word,
                            confidence_score=confidence
                        )
                        self.last_logged_word = resolved_display_word
                    result_word = resolved_display_word
            else:
                result_word = "Awaiting Two-Hand Dataset..." if is_two_handed_present else "Analyzing..."

            await self.send(text_data=json.dumps({'prediction': result_word, 'confidence': confidence}))

        except Exception as e:
            print(f"[STREAM ERROR] {e}")