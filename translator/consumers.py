import json
import numpy as np
import tensorflow as tf
from channels.generic.websocket import AsyncWebsocketConsumer

# ========================================================
# GLOBAL ENGINE INITIALIZATION
# ========================================================
# Load the model globally once on server startup so we don't
# reload a heavy model file during a live conversation.
MODEL_PATH = 'mudra_lstm_model.h5'
try:
    lstm_inference_engine = tf.keras.models.load_model(MODEL_PATH)
    print(f"[SUCCESS] Mudrā Inference Engine loaded successfully from {MODEL_PATH}")
except Exception as e:
    lstm_inference_engine = None
    print(f"[WARNING] Could not load model file natively: {e}. Running in standby mode.")

# Mudrā's corresponding output vocabulary mapping
ACTIONS = ['hello', 'thank_you', 'yes', 'no']

class SignLanguageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Called when the browser initiates a handshake connection."""
        await self.accept()
        print(f"[WS CONNECT] Client pipeline connected. Channels Channel ID: {self.channel_name}")
        
        # Send a welcome confirmation handshake back to the browser edge
        await self.send(text_data=json.dumps({
            'status': 'connected',
            'message': 'Mudra Real-Time Engine Online.'
        }))

    async def disconnect(self, close_code):
        """Called when the browser tab closes or connection breaks."""
        print(f"[WS DISCONNECT] Client pipeline severed with close code: {close_code}")
        pass

    async def receive(self, text_data):
        """Called whenever the client pushes a fresh 30-frame spatial tensor payload."""
        try:
            data = json.loads(text_data)
            
            # Extract the raw spatial coordinate matrix from the payload
            # Expects a 2D list array of shape [30, 63]
            coordinate_matrix = data.get('coordinates', None)
            
            if not coordinate_matrix:
                await self.send(text_data=json.dumps({'error': 'Missing coordinate matrix stream.'}))
                return

            # Convert to a structured NumPy matrix for tensor compatibility
            input_tensor = np.array(coordinate_matrix)
            
            # Verify the shape is exactly what our LSTM layers require: (30 frames, 63 features)
            if input_tensor.shape == (30, 63):
                # Expand dimensions to fit the batch size constraint required by TensorFlow: (1, 30, 63)
                input_tensor = np.expand_dims(input_tensor, axis=0)
                
                # Execute inference using our engine
                if lstm_inference_engine:
                    prediction_probabilities = lstm_inference_engine.predict(input_tensor, verbose=0)[0]
                    max_probability_index = np.argmax(prediction_probabilities)
                    confidence_score = float(prediction_probabilities[max_probability_index])
                    
                    # Set a baseline operational threshold to filter out random background hand twitches
                    CONFIDENCE_THRESHOLD = 0.70
                    
                    if confidence_score > CONFIDENCE_THRESHOLD:
                        predicted_word = ACTIONS[max_probability_index]
                    else:
                        predicted_word = "Analyzing..." # Unclear gesture matches
                    
                    # Push the final translation out over the full-duplex tunnel immediately
                    await self.send(text_data=json.dumps({
                        'prediction': predicted_word,
                        'confidence': confidence_score
                    }))
                else:
                    # Mock response fallback if running the script without a compiled .h5 file
                    await self.send(text_data=json.dumps({
                        'prediction': 'Standby (No Model File Located)',
                        'confidence': 0.00
                    }))
            else:
                await self.send(text_data=json.dumps({
                    'error': f'Invalid tensor dimensions. Expected (30, 63), got {input_tensor.shape}'
                }))
                
        except Exception as e:
            await self.send(text_data=json.dumps({'error': f'Inference failure: {str(e)}'}))