import json
from channels.generic.websocket import AsyncWebsocketConsumer

class TranslationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Accept the connection from the browser
        await self.accept()
        print("🟢 WebSocket Connected! Ready to receive hand data.")

    async def disconnect(self, close_code):
        print("🔴 WebSocket Disconnected.")

    async def receive(self, text_data):
        # This is where the AI coordinates will arrive from the frontend
        data = json.loads(text_data)
        landmarks = data.get('landmarks', [])

        # For now, let's just prove we are receiving the data
        if landmarks:
            print(f"Received coordinates for {len(landmarks)} hand(s)!")
            
            # Send a test response back to the browser
            await self.send(text_data=json.dumps({
                'message': 'Backend received your hand movement!'
            }))