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
    
    UPDATED: Looks up the session via its relative creation order index 
    to perfectly mirror the Option B consumer display alignment and returns
    an empty array gracefully instead of a 404 if data isn't present yet.
    """
    try:
        # 1. Gather all active session records ordered sequentially by start timestamp
        all_sessions = list(TranslationSession.objects.order_by('start_time'))
        
        # 2. Convert the incoming 1-based frontend session_id badge string into a 0-based Python array index
        target_index = int(session_id) - 1
        
        # FIXED: Instead of raising a 404 error if the session isn't in the database yet, 
        # return a successful empty history list. This stops the frontend polling errors!
        if target_index < 0 or target_index >= len(all_sessions):
            return JsonResponse({'status': 'success', 'session_id': session_id, 'history': []})
            
        # Extract the specific database record row instance mapped to that index ranking
        session = all_sessions[target_index]
        
        # Pull all related logs using the 'logs' related_name from our model
        # We use '-timestamp' to ensure the newest gestures appear at the top
        logs = session.logs.all().order_by('-timestamp')
        
        # Serialize the database rows into a clean JSON list layout
        log_data = []
        for log in logs:
            # FIXED: Using Python's native .append() structure to properly add dictionaries to your list
            log_data.append({
                'log_id': log.log_id,
                'predicted_word': log.predicted_word.capitalize() if hasattr(log.predicted_word, 'capitalize') else log.predicted_word,
                'confidence_score': f"{round(log.confidence_score * 100, 1)}%", # Formatted cleanly as a string percentage
                'timestamp': log.timestamp.strftime('%H:%M:%S')
            })
        
        return JsonResponse({'status': 'success', 'session_id': session_id, 'history': log_data})
        
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f"Internal Server Error: {str(e)}"}, status=500)

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
            # FIXED: Made sure it saves with 4-space indents for human-readability on disk
            with open(file_path, 'w') as json_file:
                json.dump(sequence, json_file, indent=4)
                
            print(f"[DATA STUDIO] Legacy JSON format dumped: {file_path}")

            return JsonResponse({
                'status': 'success', 
                'file_name': file_name, 
                'message': 'Successfully matched legacy file format.'
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'Only POST requests allowed.'}, status=405)

@csrf_exempt
def get_dataset_count(request):
    """API Endpoint: Counts how many existing JSON files match the targeted word prefix."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            label = data.get('label', '').strip().lower()
            
            if not label:
                return JsonResponse({'status': 'success', 'count': 0})
                
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
            
            if not os.path.exists(DATASET_DIR):
                return JsonResponse({'status': 'success', 'count': 0})
                
            # Count files matching the format: label_XXXXXXXXXX.json
            all_files = os.listdir(DATASET_DIR)
            matching_files = [
                f for f in all_files 
                if f.startswith(f"{label}_") and f.endswith('.json')
            ]
            
            return JsonResponse({'status': 'success', 'count': len(matching_files)})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'POST required'}, status=405)


# === NEW: AGGREGATED CORPUS COUNT API FOR SIDEBAR REGISTRY ===
def get_all_counts(request):
    """
    Scans the dataset folder and returns an aggregated list of 
    all recorded words and their total sequence counts.
    """
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
        counts = {}
        
        if os.path.exists(DATASET_DIR):
            files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
            for file in files:
                if '_' in file:
                    prefix = file.split('_')[0]
                    counts[prefix] = counts.get(prefix, 0) + 1
                    
        # Sort alphabetically for a clean UI presentation layout
        sorted_counts = dict(sorted(counts.items()))
        return JsonResponse({'status': 'success', 'counts': sorted_counts})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)