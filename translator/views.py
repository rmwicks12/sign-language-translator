from django.shortcuts import render
from django.http import JsonResponse
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