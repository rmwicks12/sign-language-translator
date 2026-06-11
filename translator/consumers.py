import os
import json
import numpy as np
import tensorflow as tf
from channels.generic.websocket import AsyncWebsocketConsumer

# ========================================================
# MACHINE LEARNING ENGINE INITIALIZATION
# ========================================================
MODEL_PATH = 'mudra_lstm_model.h5'
ACTIONS = np.array(['one', 'two', 'three'])
CONFIDENCE_THRESHOLD = 0.70  # Only print word if AI is over 70% sure

print(">> Booting Mudrā Live Inference Layers...")
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[SUCCESS] Mudrā Inference Engine loaded successfully from real dataset weights.")
    except Exception as e:
        print(f"[CRITICAL] Error initializing model file: {e}")
        model = None
else:
    print(f"[ERROR] Engine weights missing at '{MODEL_PATH}'. Run train_model.py first.")
    model = None

class SignLanguageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Accept incoming browser tracking pipeline directly."""
        await self.accept()
        print(f"WebSocket Connected: Client tracking stream established.")

    async def disconnect(self, close_code):
        """Gracefully close the frame stream."""
        print(f"WebSocket Disconnected: Client pipeline closed (Code: {close_code}).")

    async def receive(self, text_data):
        """Process the 30-frame coordinate window sent from your browser webcam."""
        if model is None:
            await self.send(text_data=json.dumps({
                'prediction': 'Engine Offline',
                'confidence': 0.0
            }))
            return

        try:
            data = json.loads(text_data)
            sequence = data.get('coordinates', [])

            # Ensure we received a complete 30-frame matrix window
            if len(sequence) == 30:
                # Shape incoming data to exactly match the network's input requirements: (1, 30, 63)
                input_data = np.expand_dims(sequence, axis=0)
                
                # Execute the forward pass prediction
                prediction_scores = model.predict(input_data, verbose=0)[0]
                best_match_idx = np.argmax(prediction_scores)
                confidence = float(prediction_scores[best_match_idx])

                # Filter out low-confidence classifications
                if confidence >= CONFIDENCE_THRESHOLD:
                    result_word = ACTIONS[best_match_idx]
                else:
                    result_word = "Analyzing..."

                # Fire the classification back up to your browser screen interface!
                await self.send(text_data=json.dumps({
                    'prediction': result_word,
                    'confidence': confidence
                }))
                
        except Exception as e:
            # Prevent the server from crashing on an anomalous packet frame
            print(f"[STREAM ERROR] Error evaluating tracking frame packet: {e}")