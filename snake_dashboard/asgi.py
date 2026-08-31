"""
ASGI config for snake_dashboard project.

Exposes the ASGI callable as a module-level variable named ``application``.
Routes plain HTTP to Django as usual and WebSocket connections to the
dashboard app's consumers (see dashboard/routing.py).
"""

import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'snake_dashboard.settings')

import django

django.setup()

from django.conf import settings
from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
from django.core.asgi import get_asgi_application

django_asgi_app = get_asgi_application()
# Uvicorn (unlike Daphne) doesn't auto-serve STATIC_URL in dev, so wrap the
# HTTP app the same way Django's own runserver would when DEBUG is on.
if settings.DEBUG:
    django_asgi_app = ASGIStaticFilesHandler(django_asgi_app)

from channels.routing import ProtocolTypeRouter, URLRouter

import dashboard.routing

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': URLRouter(dashboard.routing.websocket_urlpatterns),
})
