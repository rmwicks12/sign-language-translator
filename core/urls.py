from django.contrib import admin
from django.urls import path
from translator import views  # <--- Import your new view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # <--- Tell Django to load the homepage
]