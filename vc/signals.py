from django.core.cache import cache
from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from .views import get_embeddings


@receiver(user_logged_in)
def user_logged_in_callback(sender, request, user, **kwargs):
    embeddings = get_embeddings()
    cache.set("embeddings", embeddings, None)
