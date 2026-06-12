from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('translator.urls')),  # Seamlessly routes to translator/urls.py
]