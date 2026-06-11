import os
import json
import numpy as np
import tensorflow as tf
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.utils import timezone
from .models import TranslationSession, TranslationLog

# ========================================================
# MACHINE LEARNING ENGINE INITIALIZATION
# ========================================================
MODEL_PATH = 'mudra_lstm_model.h5'
ACTIONS = np.array(['one', 'two', 'three'])
CONFIDENCE_THRESHOLD = 0.75  # 75% confidence before logging

print(">> Booting Mudrā Live Inference Layers...")
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[SUCCESS] Mudrā Inference Engine loaded successfully.")
    except Exception as e:
        print(f"[CRITICAL] Error initializing model file: {e}")
        model = None
else:
    print(f"[ERROR] Engine weights missing at '{MODEL_PATH}'.")
    model = None

class SignLanguageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Triggers when the browser loads the page and opens the camera socket."""
        await self.accept()
        
        # Open a brand new translation session in PostgreSQL
        self.session = await self.create_translation_session()
        self.last_logged_word = None
        
        print(f"[DB SESSION] Started Session #{self.session.pk}")

        # Instantly send the active session ID back to the frontend dashboard layout
        await self.send(text_data=json.dumps({
            'session_id': self.session.pk
        }))

    async def disconnect(self, close_code):
        """Triggers when the user closes the tab or stops the camera."""
        if hasattr(self, 'session') and self.session:
            await self.close_translation_session(self.session)
            print(f"[DB SESSION] Finalized and saved Session #{self.session.pk}")
        print("WebSocket Disconnected cleanly.")

    async def receive(self, text_data):
        """Processes live landmark matrices and commits high-confidence predictions to Postgres."""
        if model is None:
            await self.send(text_data=json.dumps({'prediction': 'Engine Offline', 'confidence': 0.0}))
            return

        try:
            data = json.loads(text_data)
            sequence = data.get('coordinates', [])

            # FLICKER & EMPTY FRAME PROTECTION:
            # If the frame tracking sequence drops or is empty, reset the tracker strings safely
            if len(sequence) < 30:
                self.last_logged_word = None
                return

            # Proceed with running inference if we have a complete 30-frame matrix window
            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0)
                prediction_scores = model.predict(input_data, verbose=0)[0]
                best_match_idx = np.argmax(prediction_scores)
                confidence = float(prediction_scores[best_match_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    result_word = ACTIONS[best_match_idx]
                    
                    # Only create a permanent row if the gesture is a new distinct word event
                    if hasattr(self, 'last_logged_word') and result_word != self.last_logged_word:
                        await self.save_log_to_db(self.session, result_word, confidence)
                        self.last_logged_word = result_word
                else:
                    result_word = "Analyzing..."

                # Push the live inference result back up to the browser interface
                await self.send(text_data=json.dumps({
                    'prediction': result_word,
                    'confidence': confidence
                }))
                
        except Exception as e:
            print(f"[STREAM ERROR] Error evaluating tracking frame packet: {e}")

    # ========================================================
    # ASYNC DATABASE INTERFACE METHODS (Django ORM Workers)
    # ========================================================
    @database_sync_to_async
    def create_translation_session(self):
        """Creates a new continuous stream marker record in PostgreSQL."""
        return TranslationSession.objects.create(notes="Live Interface Translation Stream")

    @database_sync_to_async
    def close_translation_session(self, session):
        """Timestamps the end of the streaming session."""
        session.end_time = timezone.now()
        session.save()

    @database_sync_to_async
    def save_log_to_db(self, session, word, score):
        """Injects a distinct high-confidence classification record into PostgreSQL."""
        TranslationLog.objects.create(
            session=session,
            predicted_word=word,
            confidence_score=score
        )
        print(f"[DB INSERT] Logged '{word}' (Conf: {score:.2f}) to Postgres.")