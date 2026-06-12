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
        
        # FIXED: Global boot call moved safely inside the active socket connection thread
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

            # Feed spatial matrix directly into hot-swapped LSTM layers
            input_data = np.expand_dims(sequence, axis=0)
            prediction_scores = model.predict(input_data, verbose=0)[0]
            best_match_idx = np.argmax(prediction_scores)
            confidence = float(prediction_scores[best_match_idx])

            # === ADD THESE TEMPORARY DEBUG LOGS HERE ===
            print(f"[DEBUG LOG] Best Match Index: {best_match_idx} | Active Actions List: {ACTIONS}")
            print(f"[DEBUG LOG] Confidence: {confidence:.2f} | Last Logged Word State: {self.last_logged_word}")

            # Logging Engine Rules verification
            if confidence >= 0.65 and best_match_idx < len(ACTIONS):
                result_word = ACTIONS[best_match_idx]
                print(f"[DEBUG LOG] Resolved Word: '{result_word}'")
                
                if result_word != self.last_logged_word and result_word != 'awaiting_data':
                    print(f"[DB WRITE] Saving '{result_word}' to Database row...")
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