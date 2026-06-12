import os
import json
import time  # <-- ADD THIS MISSING IMPORT RIGHT HERE
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import TranslationSession, TranslationLog

def index(request):
    """Renders the main Mudrā translation interface."""
    return render(request, 'index.html')

def get_session_history(request, session_id):
    """
    API Endpoint: Fetches all high-confidence gesture logs 
    for a specific session sorted by the most recent first.
    """
    try:
        # Query PostgreSQL for the session, making sure it exists
        session = TranslationSession.objects.get(pk=session_id)
        
        # Pull all related logs using the 'logs' related_name from our model
        # We use '-timestamp' to ensure the newest gestures appear at the top
        logs = session.logs.all().order_by('-timestamp')
        
        # Serialize the database rows into a clean JSON list layout
        log_data = [
            {
                'log_id': log.log_id,
                'predicted_word': log.predicted_word,
                'confidence_score': round(log.confidence_score * 100, 1),
                'timestamp': log.timestamp.strftime('%H:%M:%S')
            }
            for log in logs
        ]
        
        return JsonResponse({'status': 'success', 'session_id': session_id, 'history': log_data})
        
    except TranslationSession.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Session not found'}, status=404)

def data_collection_view(request):
    """Renders the standalone workspace for recording new hand gesture matrices."""
    return render(request, 'translator/data_collection.html')


@csrf_exempt
def save_captured_sequence(request):
    """Saves a 30-frame coordinate sequence matching the legacy flat JSON timestamp layout."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            label = data.get('label', '').strip().lower()
            sequence = data.get('sequence', [])

            if not label or len(sequence) != 30:
                return JsonResponse({'status': 'error', 'message': 'Invalid label or frame length.'}, status=400)

            # Target directory definition (Resolving your exact flat 'dataset' root folder)
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
            os.makedirs(DATASET_DIR, exist_ok=True)

            # Generate a unique Unix timestamp matching your existing file format
            # Example output: four_1781175920.json
            timestamp_id = int(time.time())
            file_name = f"{label}_{timestamp_id}.json"
            file_path = os.path.join(DATASET_DIR, file_name)

            # Write the raw coordinate matrices structure into a plain text JSON file
            with open(file_path, 'w') as json_file:
                json.dump(sequence, json_file)
                
            print(f"[DATA STUDIO] Legacy JSON format dumped: {file_path}")

            return JsonResponse({
                'status': 'success', 
                'file_name': file_name, 
                'message': 'Successfully matched legacy file format.'
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'Only POST requests allowed.'}, status=405)