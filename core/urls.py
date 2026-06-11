from django.contrib import admin
from django.urls import path
from translator.views import index, get_session_history  # Updated to import both views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='home'),  # Routes the root URL to the frontend dashboard
    
    # New enterprise API route for historical log retrieval
    path('api/history/<int:session_id>/', get_session_history, name='session_history'),
]