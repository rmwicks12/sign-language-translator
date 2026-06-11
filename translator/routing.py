from django.urls import re_path
from . import consumers

# This is the WebSocket equivalent of standard Django urlpatterns
websocket_urlpatterns = [
    # We use ws/ as a prefix to clearly separate WebSockets from standard http traffic
    re_path(r'ws/translate/$', consumers.SignLanguageConsumer.as_asgi()),
]