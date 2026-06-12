from django.contrib import admin
from django.urls import path
from translator.views import index, get_session_history, data_collection_view, save_captured_sequence

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='home'),
    path('api/history/<int:session_id>/', get_session_history, name='session_history'),
    path('collect/', data_collection_view, name='data_collection'),
    
    # New backend API to stream coordinate arrays directly to local disk folders
    path('api/save-sequence/', save_captured_sequence, name='save_sequence'),
]