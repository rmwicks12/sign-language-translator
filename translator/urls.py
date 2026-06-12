from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('collect/', views.data_collection_view, name='data_collection'),
    path('api/history/<int:session_id>/', views.get_session_history, name='get_session_history'),
    path('api/save-sequence/', views.save_captured_sequence, name='save_sequence'),
    path('api/get-count/', views.get_dataset_count, name='get_count'),
    path('api/get-all-counts/', views.get_all_counts, name='get_all_counts'),
    
    # NEW SAFETY PATH: Routes the deletion command to the disk controller
    path('api/delete-sample/', views.delete_last_sample, name='delete_sample'),
]