import os
import json
import numpy as np
import tensorflow as tf
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .utils import get_dynamic_actions

# Resolve absolute paths safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'mudra_lstm_model.h5')

# Core Global Memory States
model = None
ACTIONS = []
last_loaded_timestamp = 0  # Tracks the modification time of the file in memory

def hot_load_model_if_updated():
    """
    Checks the hard drive to see if a background training script 
    has updated the model file. If so, it hot-swaps it live in RAM.
    """
    global model, ACTIONS, last_loaded_timestamp
    
    if not os.path.exists(MODEL_PATH):
        print("[ENGINE ERROR] mudra_lstm_model.h5 weights file not found on disk.")
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

# Initial boot-up configuration loading
hot_load_model_if_updated()


class TranslationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        
        # Ensure model variables are checked right as a socket channel links up
        await sync_to_async(hot_load_model_if_updated)()
        
        # Instantiate the new session row securely
        from .models import TranslationSession
        self.session = await sync_to_async(TranslationSession.objects.create)()
        self.last_logged_word = None
        
        # === DYNAMIC DISPLAY INDEX FIX (Option B) ===
        # Count exactly how many total session rows exist in the table right now
        active_session_index = await sync_to_async(TranslationSession.objects.count)()
        
        await self.send(text_data=json.dumps({
            'session_id': active_session_index if active_session_index > 0 else 1
        }))

    async def disconnect(self, close_code):
        if hasattr(self, 'session') and self.session:
            from django.utils import timezone
            self.session.end_time = timezone.now()
            await sync_to_async(self.session.save)()
            
            # Dynamically recalculate logging message scale index to avoid display confusion
            active_session_index = await sync_to_async(TranslationSession.objects.count)()
            print(f"[DB SESSION] Finalized and saved Session #{active_session_index}")

    async def receive(self, text_data):
        global model, ACTIONS
        
        try:
            data = json.loads(text_data)
            command = data.get('command')
            
            # Handle standard session interface state switches
            if command == "end_session":
                if hasattr(self, 'session') and self.session:
                    from django.utils import timezone
                    self.session.end_time = timezone.now()
                    await sync_to_async(self.session.save)()
                return

            if command == "start_new_session":
                from .models import TranslationSession
                self.session = await sync_to_async(TranslationSession.objects.create)()
                self.last_logged_word = None
                
                # === DYNAMIC DISPLAY INDEX FIX (Option B) ===
                active_session_index = await sync_to_async(TranslationSession.objects.count)()
                
                await self.send(text_data=json.dumps({
                    'session_id': active_session_index if active_session_index > 0 else 1
                }))
                return

            # Frame Processing Engine Block
            sequence = data.get('coordinates', [])
            if len(sequence) != 30:
                return

            # === LIVE ENGINE CHECK ===
            # Periodically check disk timestamp safely mid-stream
            await sync_to_async(hot_load_model_if_updated)()

            if model is None:
                await self.send(text_data=json.dumps({'prediction': 'Engine Offline', 'confidence': 0.0}))
                return

            # Feed spatial matrix directly into hot-swapped LSTM layers
            input_data = np.expand_dims(sequence, axis=0)
            prediction_scores = model.predict(input_data, verbose=0)[0]
            best_match_idx = np.argmax(prediction_scores)
            confidence = float(prediction_scores[best_match_idx])

            # Ensure index fallback match safety against old arrays mapping runs
            if confidence >= 0.70 and best_match_idx < len(ACTIONS):
                result_word = ACTIONS[best_match_idx]
                
                if result_word != self.last_logged_word:
                    from .models import TranslationLog
                    await sync_to_async(TranslationLog.objects.create)(
                        session=self.session,
                        predicted_word=result_word,
                        confidence_score=confidence
                    )
                    self.last_logged_word = result_word
            else:
                result_word = "Analyzing..."

            await self.send(text_data=json.dumps({
                'prediction': result_word,
                'confidence': confidence
            }))

        except Exception as e:
            print(f"[STREAM ERROR] Error evaluating frame context: {e}")