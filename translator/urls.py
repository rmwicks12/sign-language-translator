from django.urls import path
from . import views  # This is correct here because views.py is in the same folder

urlpatterns = [
    path('', views.index, name='home'),
    path('collect/', views.data_collection_view, name='data_collection'),
    path('api/history/<int:session_id>/', views.get_session_history, name='get_session_history'),
    path('api/save-sequence/', views.save_captured_sequence, name='save_sequence'),
    path('api/get-count/', views.get_dataset_count, name='get_count'),
    
    # NEW ALIGNMENT ROUTE: Connects the dashboard registry list to your views engine
    path('api/get-all-counts/', views.get_all_counts, name='get_all_counts'),
]
