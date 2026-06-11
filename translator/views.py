from django.shortcuts import render

def index(request):
    """Renders the main Mudrā translation interface."""
    return render(request, 'index.html')