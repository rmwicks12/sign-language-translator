from django.urls import path
from . import views  # This is correct here because views.py is in the same folder

urlpatterns = [
    path('', views.index, name='home'),
    path('collect/', views.data_collection_view, name='data_collection'),
    path('api/history/<int:session_id>/', views.get_session_history, name='session_history'),
    path('api/save-sequence/', views.save_captured_sequence, name='save_sequence'),
    path('api/get-count/', views.get_dataset_count, name='get_count'),
]