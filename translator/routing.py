from django.urls import re_path
from . import consumers

# Change 'urlpatterns' to 'websocket_urlpatterns'
websocket_urlpatterns = [
    re_path(r'ws/translate/$', consumers.TranslationConsumer.as_asgi()),
]