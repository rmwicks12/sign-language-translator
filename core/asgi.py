import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

# IMPORTANT: Change 'translator.routing' below to 'core.routing' if you put 
# your consumers.py and routing.py inside the core folder instead!
import translator.routing 

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# The ProtocolTypeRouter inspects incoming traffic.
# If it's a standard web page load, it hands it to Django.
# If it's a WebSocket connection, it hands it to Mudra's inference engine.
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            translator.routing.websocket_urlpatterns
        )
    ),
})