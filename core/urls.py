from django.contrib import admin
from django.urls import path
from translator.views import index  # Import our new view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='home'),  # Route the root URL to the frontend
]