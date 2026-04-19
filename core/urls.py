from django.contrib import admin
from django.urls import path
from translator import views 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'), 
    path('collect/', views.collect_data, name='collect'),
    path('save_dataset/', views.save_dataset, name='save_dataset'), # <--- Add this line
]